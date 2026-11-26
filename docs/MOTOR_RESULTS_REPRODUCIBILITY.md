# Reproduced motor evaluation results

Preserved simulation evaluator samples and official export lineage.

The five matched teacher replays each satisfy verified contact, lift and released containment. The historical placement score keeps its original endpoint and has no recorded failure stage. ACT has no fresh result on the compact split. The SmolVLA offline loss comes from validation.

| Model | Cohort | Placements | First failure | Updates in job | Offline validation loss |
| --- | --- | ---: | --- | ---: | ---: |
| ACT historical | historical ten-world | 0/10 | unrecorded (10) | 8 | 1.000461 |
| SmolVLA historical | historical ten-world | 0/10 | unrecorded (10) | 2 | 1.707521 |
| ACT aligned fit | matched fitting | 0/5 | approach (5) | 600 | 0.184860 |
| ACT clock-conditioned fit | matched fitting | 0/5 | grasp (5) | 200 | 0.192890 |
| SmolVLA compact | matched validation | 0/20 | approach (20) | 85 | 0.442980 |
| SmolVLA compact | reserved fresh | 0/20 | approach (20) | 85 | — |
