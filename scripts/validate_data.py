import json
import sys

from incident_risk.config import ROOT, load_config
from incident_risk.data import load_raw, sha256_of, validate_schema

cfg = load_config()
df = load_raw(cfg)
problems = validate_schema(df, cfg)
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data/checksums.json").write_text(
    json.dumps({cfg["data"]["path"]: sha256_of(ROOT / cfg["data"]["path"])}, indent=2)
)
print(f"rows={len(df)} cols={df.shape[1]}")
print(df.isna().mean().round(4).to_string())
print("Target rate:", df[cfg["data"]["target"]].mean().round(4))
if problems:
    print("PROBLEMS:", *problems, sep="\n - ")
    sys.exit(1)
print("Data OK")
