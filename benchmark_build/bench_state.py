"""Shared time-respecting state for T1 baselines (released tables only), on GPU.

Stream(B).queries(rows) iterates query event rows in time order. For each query it yields the target, the candidate mask
(eligible, first_seen_t < t, minus the cascade's earlier participants) and per-participant feature vectors computed
from events strictly before t and edges created strictly before t:

  pop      causal participations before t (any split)        rec      log1p hours since last participation (never -> 12)
  never    1 if no participation before t                     aff      earlier participations in cascades of the same origin
  exp      earlier participants of this cascade with an edge to the candidate
  exp1h    the same, each weighted exp(-(t - t_v) / 1 h)    exp1d    weighted exp(-(t - t_v) / 1 d)
  dout     out-degree (edges created before the split start)  din      in-degree (same)

Counts are log1p-transformed. Queries must be passed in time order.
With self_exc_scales (seconds), extra features self<s>: sum over the candidate's own earlier participations of exp(-(t - t_i)/s)
(the self-excitation term of a multivariate Hawkes process), appended after FEATURES.
"""
import duckdb
import numpy as np
import torch

FEATURES = ["pop", "rec", "never", "aff", "exp", "exp1h", "exp1d", "dout", "din"]


class Stream:
    def __init__(self, B, dev, threads=48, self_exc_scales=()):
        self.scales = list(self_exc_scales)
        con = duckdb.connect(); con.execute(f"SET threads={threads}; SET memory_limit='300GB'")
        P = f"read_parquet('{B}/participants.parquet')"
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
        ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", c.origin_id,
                             (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                             FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                             ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()
        pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
        eg = con.execute(f"SELECT src, dst, coalesce(t_created, 9e18)::BIGINT t FROM read_parquet('{B}/edges.parquet') ORDER BY src, dst").fetchnumpy()
        self.ev, self.dev, self.N = ev, dev, len(pa_["f"])
        N = self.N
        self.first = pa_["f"]; self.elig = pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)
        self.non0 = ev["position"] > 0
        self.order = np.lexsort((ev["position"], ev["cascade_id"]))
        self.cstart = np.searchsorted(ev["cascade_id"][self.order], np.arange(ev["cascade_id"].max() + 2))
        ai = np.flatnonzero(self.non0); ai = ai[np.lexsort((ev["t"][ai], ev["origin_id"][ai]))]
        self.a_t, self.a_p = ev["t"][ai], ev["participant_id"][ai]; self.astart = np.searchsorted(ev["origin_id"][ai], np.arange(N + 1))
        self.rowptr = torch.from_numpy(np.searchsorted(eg["src"], np.arange(N + 1)).astype(np.int64)).to(dev)
        self.g_dst = torch.from_numpy(eg["dst"].astype(np.int32)).to(dev); self.g_t = torch.from_numpy(eg["t"]).to(dev)
        self._eg_src, self._eg_dst, self._eg_t = eg["src"], eg["dst"], eg["t"]
        self.first_g, self.elig_g = torch.from_numpy(self.first).to(dev), torch.from_numpy(self.elig).to(dev)
        self.ev_p = torch.from_numpy(ev["participant_id"].astype(np.int64)).to(dev); self.ev_t = torch.from_numpy(ev["t"].astype(np.float64)).to(dev)
        self.ev_pos0 = torch.from_numpy(self.non0).to(dev)
        self._deg_cache = {}
        self.reset()

    def reset(self):
        self.last = torch.full((self.N,), -1.0, device=self.dev, dtype=torch.float64)
        self.pop = torch.zeros(self.N, device=self.dev, dtype=torch.float64); self.ptr = 0
        self.selfx = torch.zeros(len(self.scales), self.N, device=self.dev, dtype=torch.float64); self.t_upd = 0.0

    def degrees(self, T):
        if T not in self._deg_cache:
            m = self._eg_t < T
            self._deg_cache[T] = (torch.from_numpy(np.bincount(self._eg_src[m], minlength=self.N).astype(np.float64)).to(self.dev),
                                  torch.from_numpy(np.bincount(self._eg_dst[m], minlength=self.N).astype(np.float64)).to(self.dev))
        return self._deg_cache[T]

    def advance(self, t):
        nxt = int(np.searchsorted(self.ev["t"], t, side="left"))
        if nxt > self.ptr:
            p, tt, nz = self.ev_p[self.ptr:nxt], self.ev_t[self.ptr:nxt], self.ev_pos0[self.ptr:nxt]
            self.last.scatter_reduce_(0, p[nz], tt[nz], "amax")
            self.pop.index_add_(0, p[nz], torch.ones(int(nz.sum()), device=self.dev, dtype=torch.float64))
            self.ptr = nxt
        else:
            p = tt = None
        if self.scales:
            for k, sc in enumerate(self.scales):
                self.selfx[k] *= np.exp(-(t - self.t_upd) / sc) if self.t_upd else 0.0
                if p is not None: self.selfx[k].index_add_(0, p[nz], torch.exp(-(t - tt[nz]) / sc))
            self.t_upd = float(t)

    def query(self, j, split_start):
        """Features for event row j (call in time order). Returns (target, cand_mask, X [N, F] float32)."""
        ev, dev, N = self.ev, self.dev, self.N
        t, c, y, pos, o = int(ev["t"][j]), ev["cascade_id"][j], int(ev["participant_id"][j]), ev["position"][j], ev["origin_id"][j]
        self.advance(t)
        ix = self.order[self.cstart[c]:self.cstart[c + 1]]; pre = ix[ev["position"][ix] < pos]
        prev = torch.from_numpy(ev["participant_id"][pre].astype(np.int64)).to(dev)
        prev_t = torch.from_numpy(ev["t"][pre].astype(np.float64)).to(dev)
        cand = self.elig_g & (self.first_g < t); cand[prev] = False; cand[y] = True
        a, b = self.astart[o], self.astart[o + 1]; b = a + np.searchsorted(self.a_t[a:b], t, side="left")
        aff = torch.zeros(N, device=dev, dtype=torch.float64)
        aff.index_add_(0, torch.from_numpy(self.a_p[a:b].astype(np.int64)).to(dev), torch.ones(b - a, device=dev, dtype=torch.float64))
        s, e = self.rowptr[prev], self.rowptr[prev + 1]; ln = e - s; tot = int(ln.sum())
        ex, e1h, e1d = (torch.zeros(N, device=dev, dtype=torch.float64) for _ in range(3))
        if tot:
            base = torch.repeat_interleave(s - torch.cumsum(ln, 0) + ln, ln) + torch.arange(tot, device=dev)
            src_t = torch.repeat_interleave(prev_t, ln)
            keep = self.g_t[base] < t; d = self.g_dst[base[keep]].long(); dt = (t - src_t[keep])
            ex.index_add_(0, d, torch.ones(len(d), device=dev, dtype=torch.float64))
            e1h.index_add_(0, d, torch.exp(-dt / 3600.0)); e1d.index_add_(0, d, torch.exp(-dt / 86400.0))
        dout, din = self.degrees(split_start)
        never = (self.last < 0).double()
        rec = torch.where(self.last < 0, torch.full_like(self.last, 12.0), torch.log1p((t - self.last).clamp(min=0) / 3600.0))
        X = torch.stack([torch.log1p(self.pop), rec, never, torch.log1p(aff), torch.log1p(ex), e1h, e1d, torch.log1p(dout), torch.log1p(din)]
                        + [self.selfx[k] for k in range(len(self.scales))], 1).float()
        return y, cand, X
