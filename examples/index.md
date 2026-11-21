# Example runs

Real outputs from `python scripts/run_examples.py`.

| # | Question | Viz | Title | Trials analysed | Planner | File |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | How has the number of trials for this drug changed per year since 2015? | `time_series` | Pembrolizumab trials per year since 2015 | 2,942 of 2,942 | rule | [trend_pembrolizumab](01_trend_pembrolizumab.json) |
| 02 | How are breast cancer trials distributed across phases? | `bar_chart` | Breast cancer trials by phase | 10,000 of 17,038 | rule | [phases_breast_cancer](02_phases_breast_cancer.json) |
| 03 | Compare phases for trials involving pembrolizumab vs nivolumab | `grouped_bar_chart` | Trials by phase: pembrolizumab vs nivolumab | 5,053 of 5,053 | rule | [compare_pembrolizumab_nivolumab](03_compare_pembrolizumab_nivolumab.json) |
| 04 | Which countries have the most recruiting trials for diabetes? | `bar_chart` | Recruiting Diabetes trials by country | 1,975 of 1,975 | rule | [countries_recruiting_diabetes](04_countries_recruiting_diabetes.json) |
| 05 | Show a network of sponsors and drugs for non-small cell lung cancer trials | `network_graph` | Sponsors and drugs in Non-small cell lung cancer trials | 8,679 of 8,679 | rule | [network_sponsors_drugs_nsclc](05_network_sponsors_drugs_nsclc.json) |
| 06 | Which drugs frequently co-occur with pembrolizumab in combination studies? | `network_graph` | Drugs studied together in Pembrolizumab trials | 3,008 of 3,008 | rule | [network_drug_cooccurrence_pembrolizumab](06_network_drug_cooccurrence_pembrolizumab.json) |
| 07 | How many recruiting Alzheimer's disease trials are there? | `stat` | Number of recruiting Alzheimer's disease trials | 625 of 625 | rule | [count_recruiting_alzheimers](07_count_recruiting_alzheimers.json) |
