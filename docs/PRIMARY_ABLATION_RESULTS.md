# Registered three-robot controls

The two evidence heads share 50 prespecified independent groups. The original frozen forecasts were reused. The two hidden variants of each F6 group count as one pair. The common message ceiling is 3,072 bytes. NLL and byte means weight source groups equally.

Reversing the witness clips changes frame order and the latest frame seen by each visual head. Withholding association drops its likelihood at fusion and keeps the original transmission charge. Only the scoring oracles access evaluator or event labels.

## Event-only

| Method | First container | Resolving correct | F6 pairs inspect | Wrong variants | NLL | Mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned selective | 18/18 | 34/34 | 8/8 | 1 | 0.291 | 1537 |
| Learned broadcast | 18/18 | 34/34 | 8/8 | 1 | 0.291 | 1537 |
| One reply, validation information gain | 0/18 | 16/34 | 8/8 | 0 | 0.529 | 1155 |
| One reply, latest witness | 0/18 | 0/34 | 8/8 | 0 | 1.113 | 960 |
| Reversed witness clips | 0/18 | 16/34 | 8/8 | 26 | 2.087 | 1537 |
| Association likelihood withheld | 0/18 | 16/34 | 8/8 | 0 | 0.529 | 1537 |
| Particle filter, same learned likelihoods | 18/18 | 34/34 | 8/8 | 1 | 0.292 | 1537 |
| Scoring-only event-label oracle | 18/18 | 34/34 | 8/8 | 0 | 0.389 | 0 |
| Scoring-only terminal oracle | 18/18 | 34/34 | 0/8 | 0 | 0.000 | 0 |

## Event + reveal

| Method | First container | Resolving correct | F6 pairs inspect | Wrong variants | NLL | Mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned selective | 18/18 | 34/34 | 8/8 | 4 | 0.426 | 1537 |
| Learned broadcast | 18/18 | 34/34 | 8/8 | 4 | 0.426 | 1537 |
| One reply, validation information gain | 0/18 | 16/34 | 8/8 | 0 | 0.529 | 1155 |
| One reply, latest witness | 0/18 | 0/34 | 8/8 | 0 | 1.113 | 960 |
| Reversed witness clips | 0/18 | 16/34 | 8/8 | 16 | 1.339 | 1537 |
| Association likelihood withheld | 0/18 | 16/34 | 8/8 | 0 | 0.529 | 1537 |
| Particle filter, same learned likelihoods | 18/18 | 34/34 | 8/8 | 4 | 0.413 | 1537 |
| Scoring-only event-label oracle | 18/18 | 34/34 | 8/8 | 0 | 0.389 | 0 |
| Scoring-only terminal oracle | 18/18 | 34/34 | 0/8 | 0 | 0.000 | 0 |

## Transport and exact final-view matches

- Event-only had maximum joint total variation 0 after reordering, duplicating or adding late copies of the admitted messages. The matched particle filter differed from exact fusion by at most 0.0166. There were 0 resolving pairs with exactly equal final receiver RGB, distinct witness histories and different correct answers.
- Event + reveal had maximum joint total variation 0 after reordering, duplicating or adding late copies of the admitted messages. The matched particle filter differed from exact fusion by at most 0.0124. There were 0 resolving pairs with exactly equal final receiver RGB, distinct witness histories and different correct answers.
