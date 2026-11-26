# Fixed-profile calibration results

This is a descriptive replay of the calibration cohort. The same 48 independent source groups appear at each team size. The archive contains 168 recorded team variants.

Correct container counts the registered resolvable scenes and charges an inspection as a missed first choice. F6 inspect counts hidden variants, with two variants per source group. Answer and wrong rates, proper scores, decision cost and serialized bytes weight source groups equally within each team size. Raw RGB payload counts compressed frame bytes. Compact and broadcast rows count their declared messages.

### N = 3

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/16 | 16/16 | 2.1 | 0.0 | 1.087 | 0.656 | 0.245 | 0 |
| Event-only | Receiver private history | 0/16 | 16/16 | 0.0 | 0.0 | 0.956 | 0.561 | 0.250 | 0 |
| Event-only | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.481 | 0.238 | 0.083 | 1082 |
| Event-only | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 1082 |
| Event-only | Central full RGB | 16/16 | 16/16 | 72.9 | 0.0 | 0.362 | 0.235 | 0.068 | 470251 |
| Event-only | Central compact | 16/16 | 16/16 | 75.0 | 0.0 | 0.290 | 0.188 | 0.062 | 1531 |
| Event-only | Learned selective | 16/16 | 16/16 | 75.0 | 0.0 | 0.290 | 0.188 | 0.062 | 1531 |
| Event + reveal | Receiver current RGB | 0/16 | 16/16 | 2.1 | 0.0 | 1.087 | 0.656 | 0.245 | 0 |
| Event + reveal | Receiver private history | 0/16 | 16/16 | 0.0 | 0.0 | 0.956 | 0.561 | 0.250 | 0 |
| Event + reveal | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.481 | 0.238 | 0.083 | 1082 |
| Event + reveal | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 1082 |
| Event + reveal | Central full RGB | 16/16 | 16/16 | 83.3 | 8.3 | 0.469 | 0.275 | 0.125 | 470251 |
| Event + reveal | Central compact | 16/16 | 16/16 | 83.3 | 8.3 | 0.437 | 0.255 | 0.125 | 1531 |
| Event + reveal | Learned selective | 16/16 | 16/16 | 83.3 | 8.3 | 0.437 | 0.255 | 0.125 | 1531 |

### N = 4

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/16 | 16/16 | 2.1 | 0.0 | 1.087 | 0.656 | 0.245 | 0 |
| Event-only | Receiver private history | 0/16 | 16/16 | 0.0 | 0.0 | 0.956 | 0.561 | 0.250 | 0 |
| Event-only | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.479 | 0.238 | 0.083 | 1304 |
| Event-only | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 1304 |
| Event-only | Central full RGB | 16/16 | 16/16 | 75.0 | 0.0 | 0.362 | 0.235 | 0.062 | 486368 |
| Event-only | Central compact | 0/16 | 16/16 | 33.3 | 0.0 | 0.523 | 0.355 | 0.167 | 1705 |
| Event-only | Learned selective | 16/16 | 16/16 | 75.0 | 0.0 | 0.290 | 0.188 | 0.062 | 1705 |
| Event + reveal | Receiver current RGB | 0/16 | 16/16 | 2.1 | 0.0 | 1.087 | 0.656 | 0.245 | 0 |
| Event + reveal | Receiver private history | 0/16 | 16/16 | 0.0 | 0.0 | 0.956 | 0.561 | 0.250 | 0 |
| Event + reveal | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.479 | 0.238 | 0.083 | 1304 |
| Event + reveal | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 1304 |
| Event + reveal | Central full RGB | 15/16 | 16/16 | 64.6 | 0.0 | 0.415 | 0.255 | 0.089 | 486368 |
| Event + reveal | Central compact | 0/16 | 16/16 | 33.3 | 0.0 | 0.522 | 0.354 | 0.167 | 1705 |
| Event + reveal | Learned selective | 16/16 | 16/16 | 83.3 | 8.3 | 0.437 | 0.255 | 0.125 | 1705 |

### N = 8

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/16 | 16/16 | 6.2 | 0.0 | 1.049 | 0.632 | 0.234 | 0 |
| Event-only | Receiver private history | 0/16 | 16/16 | 6.2 | 0.0 | 0.890 | 0.520 | 0.234 | 0 |
| Event-only | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.477 | 0.238 | 0.083 | 2547 |
| Event-only | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 2547 |
| Event-only | Central full RGB | 16/16 | 0/16 | 85.4 | 8.3 | 0.394 | 0.236 | 0.120 | 545873 |
| Event-only | Central compact | 0/16 | 16/16 | 33.3 | 0.0 | 0.522 | 0.354 | 0.167 | 2461 |
| Event-only | Learned selective | 16/16 | 16/16 | 75.0 | 0.0 | 0.290 | 0.188 | 0.062 | 2411 |
| Event + reveal | Receiver current RGB | 0/16 | 16/16 | 6.2 | 0.0 | 1.049 | 0.632 | 0.234 | 0 |
| Event + reveal | Receiver private history | 0/16 | 16/16 | 6.2 | 0.0 | 0.890 | 0.520 | 0.234 | 0 |
| Event + reveal | Deterministic broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.477 | 0.238 | 0.083 | 2547 |
| Event + reveal | Calibrated particle broadcast | 16/16 | 16/16 | 66.7 | 0.0 | 0.407 | 0.250 | 0.083 | 2547 |
| Event + reveal | Central full RGB | 16/16 | 16/16 | 72.9 | 4.2 | 0.423 | 0.259 | 0.109 | 545873 |
| Event + reveal | Central compact | 0/16 | 16/16 | 33.3 | 0.0 | 0.534 | 0.365 | 0.167 | 2461 |
| Event + reveal | Learned selective | 16/16 | 16/16 | 83.3 | 8.3 | 0.437 | 0.255 | 0.125 | 2411 |
