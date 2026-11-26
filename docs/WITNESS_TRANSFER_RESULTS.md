# Witness acquisition on reserved MuJoCo viewpoints

The test contains 6 independent groups and 13 paired camera variants.

| Evidence head | Reply window | Acquisition | Correct | Incorrect | Inspect | Mean bytes | Mean decision cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| events only | 90 ms | frozen factor coverage | 7 | 3 | 3 | 1922.1 | 0.299 |
| events only | 90 ms | calibrated fixed selection | 4 | 0 | 9 | 1922.1 | 0.167 |
| events only | 90 ms | empirical expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events only | 90 ms | learned channel expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events only | 120 ms | frozen factor coverage | 7 | 3 | 3 | 1922.1 | 0.299 |
| events only | 120 ms | calibrated fixed selection | 4 | 0 | 9 | 1922.1 | 0.167 |
| events only | 120 ms | empirical expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events only | 120 ms | learned channel expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events plus reveal | 90 ms | frozen factor coverage | 9 | 4 | 0 | 1922.1 | 0.333 |
| events plus reveal | 90 ms | calibrated fixed selection | 4 | 0 | 9 | 1922.1 | 0.167 |
| events plus reveal | 90 ms | empirical expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events plus reveal | 90 ms | learned channel expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events plus reveal | 120 ms | frozen factor coverage | 9 | 4 | 0 | 1922.1 | 0.333 |
| events plus reveal | 120 ms | calibrated fixed selection | 4 | 0 | 9 | 1922.1 | 0.167 |
| events plus reveal | 120 ms | empirical expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |
| events plus reveal | 120 ms | learned channel expected risk | 4 | 0 | 9 | 1467.1 | 0.167 |

Mean costs and bytes weight each source group equally.
The paired bootstrap intervals use the underlying source group as the unit. The small test is descriptive.
