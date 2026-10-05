# LEMMA QA results (2x1 + 2x2 takes)

**Table 1.** Accuracy (%) of Qwen3-VL per question type (Q1-Q8, see Table 2) under each view condition on LEMMA 2x1 + 2x2 takes. The first row gives the number of questions per type. Best per column in bold. Source: `outputs/qa_all_*.jsonl`.

| Views | Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *#Questions* | | 141 | 84 | 72 | 132 | 56 | 81 | 683 | 92 | 1341 |
| fpv1 | 8b | **58.2** | **64.3** | 56.9 | **38.6** | 60.7 | **51.9** | 51.5 | 41.3 | 51.8 |
| fpv2 | 8b | 39.7 | 52.4 | 41.7 | 34.1 | 60.7 | 48.1 | **56.5** | 29.3 | 49.3 |
| master | 8b | 44.7 | 56.0 | 40.3 | 28.8 | 57.1 | 39.5 | 48.0 | 41.3 | 45.3 |
| fpv1+fpv2 | 8b | 48.9 | 60.7 | **59.7** | 36.4 | **64.3** | **51.9** | 56.1 | 39.1 | **52.8** |
| fpv1+master | 8b | 51.8 | 58.3 | 56.9 | 33.3 | 57.1 | **51.9** | 49.0 | **43.5** | 48.9 |
| fpv2+master | 8b | 42.6 | 51.2 | 41.7 | 29.5 | 58.9 | 44.4 | 54.6 | 33.7 | 48.1 |

Comments:
- The accuracy fpv1 egocentric is consistently better than fpv2 and master view. There are two possible reasons: (a) the questions/benchmark itself is bias toward fpv1 egocentric videos, and (b) different model have different favor of ego or exo centric? 

## Question types

**Table 2.** LEMMA question types and the number of questions of each type.

| Type | Description | #Questions |
|---|---|---:|
| Q1 | Initial object location | 141 |
| Q2 | Person location during action | 84 |
| Q3 | Object destination after action | 72 |
| Q4 | Egocentric relative direction | 132 |
| Q5 | Object state change | 56 |
| Q6 | Object-object spatial relation | 81 |
| Q7 | Other person's location (multi-agent) | 683 |
| Q8 | Activity order | 92 |
