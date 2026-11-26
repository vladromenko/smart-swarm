# Confidence cutoffs for inspection

Saved posteriors for 48 calibration and 100 independent test sources. Team sizes reuse the same source groups. This table shows N=8 on the original test. Confidence equal to the threshold results in a direct answer. Coverage and the wrong share among answers give each source equal weight. Both F6 variants share one source weight.

| Objective | Method | Frozen cutoff | Coverage | Wrong variants | Resolving correct | F6 pairs inspect | Cutoff strictly above | Cutoff at most |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Event-only | Learned selective | 0.85 | 73.0% | 1 | 68/68 | 16/16 | 0.8899 | 0.9809 |
| Event-only | Visual marker check | 0.75 | 68.0% | 0 | 68/68 | 16/16 | 0.4993 | 0.9809 |
| Event-only | Particle broadcast | 0.75 | 68.0% | 0 | 68/68 | 16/16 | 0.4963 | 0.9840 |
| Event + reveal | Learned selective | 0.75 | 84.0% | 8 | 68/68 | 16/16 | 0.9427 | 0.9177 |
| Event + reveal | Visual marker check | 0.75 | 68.0% | 0 | 68/68 | 16/16 | 0.4998 | 0.9177 |
| Event + reveal | Particle broadcast | 0.75 | 68.0% | 0 | 68/68 | 16/16 | 0.4963 | 0.9840 |

The final two columns describe the bounds for retaining every resolving control while avoiding all wrong direct choices and inspecting every F6 pair. Feasibility also requires a correct argmax on each resolving control. The cutoff range is descriptive. All evaluated thresholds remain the ones selected on validation.

Dots on the coverage and risk figure identify the validation-selected operating points. The report includes paired source bootstrap intervals for the change in decision cost with the marker check.
