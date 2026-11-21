"""Content checksum of parquet tables, independent of the writer (DuckDB/Parquet version, row-group layout, file row order).

The rows are sorted by every column. Each column is serialized in a fixed form (integers and timestamps as int64,
floats as float64, booleans as uint8, strings as the sorted distinct values plus each row's index in them, nulls as a separate mask). SHA-256 runs over
the column names, the serialized columns and the masks. Two tables with the same rows and columns get the same hash.

    python content_hash.py FILE.parquet [FILE2.parquet ...]        prints "<sha256>  <rows>  <file>" per table
    python content_hash.py --manifest OUT.tsv ROOT FILE ...         also writes a manifest (paths relative to ROOT)
"""
import hashlib
import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def content_hash(path):
    t = pq.read_table(path)
    t = t.select(sorted(t.column_names))
    # 64-bit string offsets: tables over ~2^31 bytes of string data (Bluesky L edges) overflow the 32-bit type when sorted.
    # The hash covers names, validity and values only, so the change does not alter any hash (2026-11-20).
    t = t.cast(pa.schema([pa.field(f.name, pa.large_string() if pa.types.is_string(f.type) else f.type) for f in t.schema]))
    t = t.sort_by([(c, "ascending") for c in t.column_names]) if t.num_rows else t
    h = hashlib.sha256()
    for name in t.column_names:
        col = t.column(name).combine_chunks(); h.update(name.encode() + b"\0")
        valid = np.asarray(pc.is_valid(col).to_numpy(zero_copy_only=False), np.uint8); h.update(valid.tobytes())
        ty = col.type
        if pa.types.is_boolean(ty): v = np.asarray(pc.fill_null(col, False).to_numpy(zero_copy_only=False), np.uint8)
        elif pa.types.is_integer(ty) or pa.types.is_timestamp(ty) or pa.types.is_date(ty):
            v = np.asarray(pc.fill_null(col.cast(pa.int64()), 0).to_numpy(zero_copy_only=False), np.int64)
        elif pa.types.is_floating(ty): v = np.asarray(pc.fill_null(col.cast(pa.float64()), 0.0).to_numpy(zero_copy_only=False), np.float64)
        elif pa.types.is_string(ty) or pa.types.is_large_string(ty) or pa.types.is_dictionary(ty):
            s = pc.fill_null(col.cast(pa.large_string()), ""); u = pc.unique(s); u = u.take(pc.sort_indices(u))   # sorted distinct values
            h.update("\0".join(u.to_pylist()).encode() + b"\1")
            v = np.asarray(pc.index_in(s, value_set=u).to_numpy(zero_copy_only=False), np.int64)          # each row's index in them
        else: raise TypeError(f"{path}: column {name} has unsupported type {ty}")
        h.update(v.tobytes())
    return h.hexdigest(), t.num_rows


if __name__ == "__main__":
    a = sys.argv[1:]; man = None
    if a and a[0] == "--manifest": man, root, a = a[1], a[2], a[3:]
    rows = []
    for f in a:
        d, n = content_hash(f); rows.append((d, n, f)); print(f"{d}  {n}  {f}", flush=True)
    if man:
        with open(man, "w") as o:
            o.write("content_sha256\trows\tpath\n")
            for d, n, f in rows: o.write(f"{d}\t{n}\t{os.path.relpath(f, root)}\n")
