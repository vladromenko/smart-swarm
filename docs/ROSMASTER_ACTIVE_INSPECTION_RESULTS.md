# Matched moving-camera inspection

The moving and stationary branches used the same saved reset and the same two exposure times.
The original allocation had 20 independent scenes. Of 19 attempted scenes, 17 were scored.

| Scene set | Method | Moving correct | Stationary correct | Paired effect (pp) | 95% interval (pp) | All allocated bounds (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALL | RGB localizer | 15/17 | 0/17 | +88.2 | +37.3 to +99.0 | +60.0 to +90.0 |
| ALL | Events only | 16/17 | 9/17 | +41.2 | -6.5 to +70.1 | +20.0 to +50.0 |
| ALL | Events + reveal | 16/17 | 9/17 | +41.2 | -6.5 to +70.1 | +20.0 to +50.0 |
| F1 | RGB localizer | 8/9 | 0/9 | +88.9 | +8.5 to +99.9 | +70.0 to +90.0 |
| F1 | Events only | 9/9 | 9/9 | +0.0 | -38.5 to +38.5 | -10.0 to +10.0 |
| F1 | Events + reveal | 9/9 | 9/9 | +0.0 | -38.5 to +38.5 | -10.0 to +10.0 |
| F6 | RGB localizer | 7/8 | 0/8 | +87.5 | +0.3 to +99.8 | +50.0 to +90.0 |
| F6 | Events only | 7/8 | 0/8 | +87.5 | +0.3 to +99.8 | +50.0 to +90.0 |
| F6 | Events + reveal | 7/8 | 0/8 | +87.5 | +0.3 to +99.8 | +50.0 to +90.0 |

The allocation bounds assign each missing pair any possible outcome.
The endpoint is simulated location. The learned methods include direct answers they issued before motion.
