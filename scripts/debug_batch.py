import json
import urllib.error
import urllib.request

from incident_risk.config import load_config
from incident_risk.data import load_raw
from incident_risk.splits import chronological_split

cfg = load_config()
feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
train, _, _ = chronological_split(load_raw(cfg), cfg)
rows = json.loads(train[feats].dropna().sample(100, random_state=42).to_json(orient="records"))
req = urllib.request.Request(
    "http://localhost:8000/predict/batch?suppress=false",
    data=json.dumps({"events": rows}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    print(urllib.request.urlopen(req).read().decode()[:300])
except urllib.error.HTTPError as x:
    print(x.code, x.read().decode()[:800])