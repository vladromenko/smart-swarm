# Independent primary test results

This is the released independent test cohort. The same 100 independent source groups appear at each team size. The archive contains 348 recorded team variants.

Correct container counts the registered resolvable scenes and charges an inspection as a missed first choice. F6 inspect counts hidden variants, with two variants per source group. Answer and wrong rates, proper scores, decision cost and serialized bytes weight source groups equally within each team size. Raw RGB payload counts compressed frame bytes. Compact and broadcast rows count their declared messages.

The eight-agent comparison of inspection and correct first answers counts each F6 pair once and uses the 68 separate resolving controls from F1, F2, F3 and F5. It includes the always-inspect reference. It illustrates how a predictor with fully coupled F6 outputs can still make an unsupported direct answer. [Inspection and resolving-control chart](../evidence/learning-pilot-v3/primary-test-inspection-controls.pdf)

### N = 3

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.075 | 0.648 | 0.240 | 0 |
| Event-only | Receiver private history | 0/34 | 32/32 | 0.0 | 0.0 | 0.933 | 0.546 | 0.250 | 0 |
| Event-only | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.465 | 0.229 | 0.080 | 1092 |
| Event-only | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 1092 |
| Event-only | Central full RGB | 34/34 | 32/32 | 74.0 | 2.0 | 0.351 | 0.225 | 0.085 | 471372 |
| Event-only | Central compact | 34/34 | 32/32 | 73.0 | 1.0 | 0.287 | 0.183 | 0.077 | 1537 |
| Event-only | Learned selective | 34/34 | 32/32 | 73.0 | 1.0 | 0.287 | 0.183 | 0.077 | 1537 |
| Event + reveal | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.075 | 0.648 | 0.240 | 0 |
| Event + reveal | Receiver private history | 0/34 | 32/32 | 0.0 | 0.0 | 0.933 | 0.546 | 0.250 | 0 |
| Event + reveal | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.465 | 0.229 | 0.080 | 1092 |
| Event + reveal | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 1092 |
| Event + reveal | Central full RGB | 34/34 | 32/32 | 84.0 | 8.0 | 0.454 | 0.265 | 0.120 | 471372 |
| Event + reveal | Central compact | 34/34 | 32/32 | 84.0 | 8.0 | 0.424 | 0.245 | 0.120 | 1537 |
| Event + reveal | Learned selective | 34/34 | 32/32 | 84.0 | 8.0 | 0.424 | 0.245 | 0.120 | 1537 |

### N = 4

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.075 | 0.648 | 0.240 | 0 |
| Event-only | Receiver private history | 0/34 | 32/32 | 0.0 | 0.0 | 0.933 | 0.546 | 0.250 | 0 |
| Event-only | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.463 | 0.229 | 0.080 | 1327 |
| Event-only | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 1327 |
| Event-only | Central full RGB | 34/34 | 32/32 | 74.0 | 2.0 | 0.351 | 0.225 | 0.085 | 487841 |
| Event-only | Central compact | 0/34 | 32/32 | 34.0 | 0.0 | 0.524 | 0.359 | 0.165 | 1711 |
| Event-only | Learned selective | 34/34 | 32/32 | 73.0 | 1.0 | 0.287 | 0.183 | 0.077 | 1711 |
| Event + reveal | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.075 | 0.648 | 0.240 | 0 |
| Event + reveal | Receiver private history | 0/34 | 32/32 | 0.0 | 0.0 | 0.933 | 0.546 | 0.250 | 0 |
| Event + reveal | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.463 | 0.229 | 0.080 | 1327 |
| Event + reveal | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 1327 |
| Event + reveal | Central full RGB | 31/34 | 32/32 | 65.0 | 0.0 | 0.403 | 0.246 | 0.087 | 487841 |
| Event + reveal | Central compact | 0/34 | 32/32 | 34.0 | 0.0 | 0.515 | 0.350 | 0.165 | 1711 |
| Event + reveal | Learned selective | 34/34 | 32/32 | 84.0 | 8.0 | 0.424 | 0.245 | 0.120 | 1711 |

### N = 8

| Objective | Method | Correct container | F6 inspect | Answer % | Wrong % | NLL | Brier | Cost | Mean bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.063 | 0.640 | 0.240 | 0 |
| Event-only | Receiver private history | 0/34 | 32/32 | 4.0 | 0.0 | 0.918 | 0.538 | 0.240 | 0 |
| Event-only | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.461 | 0.229 | 0.080 | 2606 |
| Event-only | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 2606 |
| Event-only | Central full RGB | 34/34 | 0/32 | 86.0 | 10.0 | 0.384 | 0.229 | 0.135 | 547717 |
| Event-only | Central compact | 0/34 | 32/32 | 34.0 | 0.0 | 0.515 | 0.350 | 0.165 | 2467 |
| Event-only | Learned selective | 34/34 | 32/32 | 73.0 | 1.0 | 0.285 | 0.181 | 0.077 | 2417 |
| Event + reveal | Receiver current RGB | 0/34 | 32/32 | 4.0 | 0.0 | 1.063 | 0.640 | 0.240 | 0 |
| Event + reveal | Receiver private history | 0/34 | 32/32 | 4.0 | 0.0 | 0.918 | 0.538 | 0.240 | 0 |
| Event + reveal | Deterministic broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.461 | 0.229 | 0.080 | 2606 |
| Event + reveal | Calibrated particle broadcast | 34/34 | 32/32 | 68.0 | 0.0 | 0.391 | 0.240 | 0.080 | 2606 |
| Event + reveal | Central full RGB | 34/34 | 32/32 | 76.0 | 4.0 | 0.410 | 0.250 | 0.100 | 547717 |
| Event + reveal | Central compact | 0/34 | 32/32 | 34.0 | 0.0 | 0.533 | 0.367 | 0.165 | 2467 |
| Event + reveal | Learned selective | 34/34 | 32/32 | 84.0 | 8.0 | 0.424 | 0.245 | 0.120 | 2417 |
