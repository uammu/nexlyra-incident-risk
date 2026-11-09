# SLO proposals

| SLO | Target | Evidence |
|---|---|---|
| API availability | 99.5% monthly | Prometheus `up` metric |
| `/predict` p95 latency | under 200 ms | Docker: 65 ms single client, 89 ms at 8 clients, 183 ms at 16 clients (`reports/benchmark.json`) |
| Alert volume | 25 to 100 per 1,000 | Prometheus rule `AlertBudgetBurn` |
| Weekly precision | at least 50% once real labels exist | validation 0.93, test 0.89 |
| Drift review | within 1 business day of "investigate" | monitoring plan |