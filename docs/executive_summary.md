# Executive summary

**What it does.** Scores how likely each service event is to become a major incident within an hour, so on-call engineers review the riskiest few first.

**Result (held-out test, evaluated once).** Of every 100 alerts, about 89 were real incidents. At 50 alerts per 1,000 events it catches about 42% of incidents; the ceiling at that budget is about 47%. On validation, about 100 alerts per 1,000 would catch roughly two thirds at about 72% precision (a recommendation, not re-tested on the test set).

**What drives risk.** Latency first, then error rate, CPU and dependency failures.

**Risks.** The data is synthetic, so none of this proves real-world accuracy. A sudden latency spike would flood alerts, and many missing readings reduce accuracy.

**Recommendation.** Do not alert live yet. Validate on real, consented telemetry, then pilot at level 1 (tickets only) before paging.