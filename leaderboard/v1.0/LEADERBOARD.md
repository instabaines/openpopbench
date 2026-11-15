# OpenPopBench v1.0 leaderboard

Test Hits@100 (%), mean ± sd over seeds. Heuristic rows are deterministic. † fails the sanity gate, or is flagged by the maintainers (meta.json "flags", e.g. a failed reproduction gate). § scored on the 10% query subsample. ✓ verified (re-run by the maintainers).

## bluesky_S

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| TPNet§✓ | temporal_graph | 67.4 ± 0.5 | 43.8 ± 0.2 | 26.9 ± 3.2 |
| Learned feature ranker✓ | feature_ranker | 65.4 ± 0.9 | 22.1 ± 2.4 | 27.1 ± 0.1 |
| recency | heuristic | 52.7 | 36.7 | 24.2 |
| Exposure Hawkes✓ | other | 45.6 ± 0.1 | 15.7 ± 0.7 | 16.7 ± 0.4 |
| affinity | heuristic | 40.6 | 2.4 | 16.8 |
| HyperEvent§✓ | temporal_graph | 40.2 ± 0.1 | 21.0 ± 0.2 | 13.2 ± 2.3 |
| ReaRec✓ | sequential_recommender | 38.4 ± 0.6 | 12.1 ± 0.8 | 0.0 |
| FEARec✓ | sequential_recommender | 37.8 ± 0.3 | 13.7 ± 0.2 | 0.0 |
| MINDS✓ | diffusion | 37.1 ± 0.2 | 12.9 ± 0.1 | 0.0 |
| DCRec✓ | sequential_recommender | 36.6 ± 0.3 | 12.4 ± 0.1 | 0.0 |
| S2P2✓ | point_process | 36.4 ± 1.2 | 12.6 ± 0.5 | 0.0 |
| MIM✓ | diffusion | 35.9 ± 0.3 | 12.8 ± 0.2 | 0.0 |
| MS-HGAT✓ | diffusion | 34.5 ± 0.2 | 11.9 ± 0.0 | 0.0 |
| AttNHP✓ | point_process | 33.8 ± 0.4 | 10.7 ± 0.2 | 0.0 |
| DyHGCN✓ | diffusion | 33.4 ± 0.4 | 11.6 ± 0.0 | 0.0 |
| RMTPP✓ | point_process | 32.6 ± 0.2 | 10.9 ± 0.2 | 0.0 |
| SASRec✓ | sequential_recommender | 31.4 ± 0.6 | 10.2 ± 0.3 | 0.0 |
| WSMTHP†✓ | point_process | 19.7 ± 1.1 | 6.6 ± 0.3 | 0.0 |
| popularity | heuristic | 19.2 | 6.6 | 0.0 |
| PMRCA†✓ | diffusion | 16.9 ± 0.1 | 6.3 ± 0.3 | 0.0 |
| exposure | heuristic | 14.6 | 0.7 | 7.3 |
| EvolveGCN†✓ | temporal_graph | 14.5 ± 0.4 | 7.2 ± 0.1 | 2.9 ± 2.0 |
| TNCN†§✓ | temporal_graph | 12.4 ± 12.4 | 5.8 ± 6.4 | 2.1 ± 2.5 |

## bluesky_Mid

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| TPNet§✓ | temporal_graph | 55.9 ± 1.0 | 33.7 ± 1.2 | 49.8 ± 5.0 |
| Learned feature ranker✓ | feature_ranker | 54.0 ± 0.5 | 15.3 ± 0.6 | 47.2 ± 1.3 |
| Exposure Hawkes✓ | other | 32.8 ± 0.0 | 4.8 ± 0.2 | 27.9 ± 0.6 |
| HyperEvent§✓ | temporal_graph | 32.8 ± 0.5 | 15.7 ± 0.3 | 30.5 ± 1.3 |
| recency | heuristic | 31.4 | 21.2 | 21.9 |
| affinity | heuristic | 30.6 | 0.8 | 27.1 |
| ReaRec✓ | sequential_recommender | 28.2 ± 0.6 | 8.1 ± 0.4 | 0.0 |
| FEARec✓ | sequential_recommender | 26.0 ± 0.1 | 9.1 ± 0.1 | 0.0 |
| DCRec✓ | sequential_recommender | 25.7 ± 0.4 | 8.6 ± 0.2 | 0.0 |
| MINDS✓ | diffusion | 23.4 ± 0.2 | 7.7 ± 0.1 | 0.0 |
| SASRec✓ | sequential_recommender | 17.9 ± 0.1 | 5.8 ± 0.3 | 0.0 |
| AttNHP✓ | point_process | 16.1 ± 0.3 | 5.4 ± 0.0 | 0.0 |
| RMTPP✓ | point_process | 12.1 ± 0.1 | 4.3 ± 0.1 | 0.0 |
| exposure | heuristic | 11.0 | 0.7 | 5.0 |
| S2P2✓ | point_process | 10.6 ± 5.0 | 3.5 ± 1.4 | 0.0 |
| TNCN§✓ | temporal_graph | 9.7 ± 8.4 | 5.3 ± 4.1 | 9.8 ± 3.4 |
| popularity | heuristic | 8.8 | 3.2 | 0.0 |
| WSMTHP†✓ | point_process | 1.4 ± 1.1 | 0.7 ± 0.5 | 0.0 |

## bluesky_M

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| Learned feature ranker✓ | feature_ranker | 37.1 ± 0.7 | 14.7 ± 1.3 | 44.5 ± 1.1 |
| Exposure Hawkes✓ | other | 21.4 ± 0.1 | 2.0 ± 0.1 | 20.5 ± 0.1 |
| affinity | heuristic | 20.4 | 0.3 | 17.9 |
| ReaRec✓ | sequential_recommender | 17.5 | 6.4 | 0.0 |
| exposure | heuristic | 7.8 | 0.6 | 4.0 |
| recency | heuristic | 4.3 | 2.9 | 6.5 |
| SASRec✓ | sequential_recommender | 4.1 ± 1.3 | 2.0 ± 0.5 | 0.0 |
| RMTPP✓ | point_process | 3.9 | 1.9 | 0.0 |
| popularity | heuristic | 2.8 | 1.3 | 0.0 |
| S2P2†✓ | point_process | 1.8 | 0.9 | 0.0 |
| AttNHP†✓ | point_process | 0.0 | 0.0 | 0.0 |

## bluesky_L

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| Learned feature ranker✓ | feature_ranker | 37.9 ± 0.4 | 15.8 ± 0.7 | 33.9 ± 0.6 |
| Exposure Hawkes✓ | other | 23.8 ± 0.1 | 1.9 ± 0.1 | 17.1 ± 0.4 |
| affinity | heuristic | 21.4 | 0.2 | 12.6 |
| exposure | heuristic | 10.8 | 1.9 | 4.4 |
| recency | heuristic | 1.5 | 0.9 | 1.3 |
| popularity | heuristic | 1.3 | 0.7 | 0.0 |
| SASRec†✓ | sequential_recommender | 0.6 ± 0.2 | 0.4 ± 0.1 | 0.0 |

## telegram_L

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| Learned feature ranker✓ | feature_ranker | 84.9 ± 0.2 | 15.2 ± 2.7 | 65.7 ± 0.4 |
| Exposure Hawkes✓ | other | 81.9 ± 0.0 | 17.0 ± 0.1 | 54.5 ± 0.0 |
| affinity | heuristic | 80.6 | 6.7 | 50.1 |
| ReaRec✓ | sequential_recommender | 75.7 ± 0.7 | 38.4 ± 0.6 | 0.0 |
| MINDS✓ | diffusion | 69.0 | 30.4 | 0.0 |
| exposure | heuristic | 66.9 | 15.0 | 38.4 |
| S2P2✓ | point_process | 58.6 | 22.5 | 0.0 |
| DyHGCN✓ | diffusion | 51.2 | 18.7 | 0.0 |
| RMTPP✓ | point_process | 50.5 | 17.0 | 0.0 |
| AttNHP✓ | point_process | 48.0 | 16.9 | 0.0 |
| recency | heuristic | 25.2 | 11.2 | 20.8 |
| popularity | heuristic | 20.8 | 8.0 | 0.0 |
| WSMTHP†✓ | point_process | 20.6 | 4.9 | 0.0 |
| SASRec†✓ | sequential_recommender | 16.5 ± 1.4 | 4.3 ± 0.3 | 0.0 |

## reddit_S

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| TPNet§✓ | temporal_graph | 14.0 ± 0.5 | 9.4 ± 0.3 | 3.5 ± 0.2 |
| exposure | heuristic | 12.4 | 5.3 | 1.2 |
| Learned feature ranker✓ | feature_ranker | 12.3 ± 0.7 | 6.6 ± 0.1 | 1.2 ± 0.4 |
| MIM✓ | diffusion | 12.3 ± 0.0 | 7.1 ± 0.0 | 0.1 ± 0.0 |
| Exposure Hawkes†✓ | other | 11.7 ± 0.1 | 5.7 ± 0.0 | 2.4 ± 0.0 |
| ReaRec✓ | sequential_recommender | 11.7 ± 0.1 | 6.8 ± 0.2 | 0.0 |
| FEARec✓ | sequential_recommender | 11.4 ± 0.1 | 6.3 ± 0.1 | 0.0 |
| DyHGCN✓ | diffusion | 11.3 ± 0.1 | 6.3 ± 0.1 | 0.0 |
| HyperEvent§✓ | temporal_graph | 10.9 ± 0.2 | 5.7 ± 0.1 | 2.6 ± 0.2 |
| MS-HGAT✓ | diffusion | 10.2 ± 0.4 | 5.8 ± 0.2 | 0.0 |
| DCRec✓ | sequential_recommender | 9.6 ± 0.3 | 5.0 ± 0.1 | 0.0 |
| affinity | heuristic | 9.0 | 3.0 | 0.7 |
| SASRec✓ | sequential_recommender | 8.2 ± 0.2 | 4.5 ± 0.2 | 0.0 |
| MINDS✓ | diffusion | 6.4 ± 0.6 | 4.1 ± 0.2 | 0.0 |
| RMTPP✓ | point_process | 6.1 ± 0.1 | 4.5 ± 0.2 | 0.0 |
| recency | heuristic | 5.6 | 5.1 | 3.8 |
| AttNHP✓ | point_process | 5.5 ± 0.2 | 4.0 ± 0.2 | 0.0 |
| WSMTHP✓ | point_process | 5.4 ± 0.1 | 3.9 ± 0.2 | 0.0 |
| TNCN†§✓ | temporal_graph | 4.8 ± 1.1 | 2.9 ± 0.9 | 0.0 |
| popularity | heuristic | 4.6 | 3.4 | 0.0 |
| S2P2†✓ | point_process | 4.1 ± 0.5 | 2.9 ± 0.4 | 0.0 |
| PMRCA†✓ | diffusion | 3.9 ± 0.1 | 2.3 ± 0.2 | 0.0 |
| EvolveGCN†✓ | temporal_graph | 1.9 ± 0.2 | 1.3 ± 0.2 | 0.6 ± 0.2 |

## reddit_Mid

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| TPNet§✓ | temporal_graph | 13.4 ± 1.3 | 9.2 ± 0.6 | 5.3 ± 0.5 |
| FEARec✓ | sequential_recommender | 10.4 ± 0.2 | 6.5 ± 0.2 | 0.0 |
| Learned feature ranker✓ | feature_ranker | 10.0 ± 0.9 | 4.8 ± 0.4 | 1.3 ± 0.6 |
| ReaRec✓ | sequential_recommender | 9.7 ± 0.2 | 6.0 ± 0.2 | 0.0 |
| exposure | heuristic | 9.2 | 4.0 | 0.5 |
| DCRec✓ | sequential_recommender | 8.9 ± 0.2 | 5.1 ± 0.2 | 0.0 |
| Exposure Hawkes†✓ | other | 8.0 ± 0.1 | 3.6 ± 0.0 | 1.4 |
| HyperEvent§✓ | temporal_graph | 7.8 ± 0.5 | 3.6 ± 0.2 | 1.1 ± 0.5 |
| MINDS✓ | diffusion | 5.9 ± 0.2 | 3.4 ± 0.1 | 0.0 |
| affinity | heuristic | 5.6 | 1.2 | 0.0 |
| SASRec✓ | sequential_recommender | 5.4 ± 0.3 | 3.1 ± 0.2 | 0.0 |
| MS-HGAT✓ | diffusion | 4.5 ± 0.2 | 3.0 ± 0.1 | 0.0 |
| AttNHP✓ | point_process | 3.4 ± 1.0 | 2.3 ± 0.3 | 0.0 |
| RMTPP✓ | point_process | 3.1 ± 0.0 | 2.4 ± 0.0 | 0.0 |
| PMRCA✓ | diffusion | 3.0 ± 0.1 | 1.8 ± 0.2 | 0.0 |
| DyHGCN✓ | diffusion | 2.5 ± 0.0 | 1.9 ± 0.0 | 0.0 |
| recency | heuristic | 2.1 | 2.0 | 1.2 |
| popularity | heuristic | 2.1 | 1.6 | 0.0 |
| WSMTHP†✓ | point_process | 1.7 ± 0.9 | 1.3 ± 0.6 | 0.0 |
| S2P2†✓ | point_process | 1.6 ± 0.1 | 1.2 ± 0.1 | 0.0 |
| EvolveGCN†✓ | temporal_graph | 1.2 ± 0.1 | 0.9 ± 0.1 | 0.3 ± 0.3 |
| TNCN†§✓ | temporal_graph | 1.1 ± 0.8 | 0.9 ± 0.7 | 0.0 |

## reddit_M

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| TPNet§✓ | temporal_graph | 13.1 | 10.3 | 8.0 |
| Learned feature ranker✓ | feature_ranker | 8.9 ± 0.5 | 5.3 ± 0.2 | 3.8 ± 0.4 |
| ReaRec✓ | sequential_recommender | 8.6 | 6.0 | 0.0 |
| FEARec✓ | sequential_recommender | 8.1 | 5.9 | 0.0 |
| exposure | heuristic | 7.7 | 3.7 | 0.9 |
| HyperEvent§✓ | temporal_graph | 6.9 | 5.1 | 2.7 |
| Exposure Hawkes†✓ | other | 5.7 ± 0.1 | 2.8 ± 0.1 | 4.5 ± 0.2 |
| affinity | heuristic | 3.5 | 0.5 | 0.2 |
| SASRec✓ | sequential_recommender | 3.2 ± 0.3 | 2.0 ± 0.1 | 0.0 |
| MINDS✓ | diffusion | 2.4 | 1.6 | 0.0 |
| PMRCA✓ | diffusion | 1.1 | 0.8 | 0.0 |
| S2P2✓ | point_process | 1.0 | 0.8 | 0.0 |
| RMTPP✓ | point_process | 1.0 | 0.7 | 0.0 |
| popularity | heuristic | 0.9 | 0.7 | 0.0 |
| AttNHP†✓ | point_process | 0.9 | 0.6 | 0.0 |
| WSMTHP†✓ | point_process | 0.8 | 0.6 | 0.0 |
| EvolveGCN†✓ | temporal_graph | 0.7 | 0.5 | 0.9 |
| recency | heuristic | 0.4 | 0.3 | 0.9 |
| TNCN†§✓ | temporal_graph | 0.3 | 0.1 | 0.0 |

## reddit_L

| method | family | all events | no-tie events | newcomers (Task 2b) |
|---|---|---|---|---|
| Learned feature ranker✓ | feature_ranker | 7.1 ± 0.3 | 4.8 ± 0.5 | 1.7 ± 0.6 |
| exposure | heuristic | 6.3 | 3.8 | 1.0 |
| Exposure Hawkes†✓ | other | 3.5 ± 0.1 | 2.9 ± 0.0 | 3.1 ± 0.1 |
| affinity | heuristic | 2.7 | 0.6 | 0.6 |
| ReaRec✓ | sequential_recommender | 1.0 | 0.9 | 0.0 |
| popularity | heuristic | 0.9 | 0.7 | 0.0 |
| SASRec†✓ | sequential_recommender | 0.7 ± 0.1 | 0.6 ± 0.1 | 0.0 |
| recency | heuristic | 0.3 | 0.1 | 0.2 |

