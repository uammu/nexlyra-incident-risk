CSV → validate → chronological split → train (5+2 models) → calibrate → champion
   → register (checksum) → API (/predict, /metrics) → Prometheus → Grafana
                         → dashboard      → drift report → retraining trigger