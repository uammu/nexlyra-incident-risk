import json
import logging
import os
import time

from incident_risk.config import ROOT

_path = os.getenv("AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl"))
os.makedirs(os.path.dirname(_path), exist_ok=True)

_logger = logging.getLogger("audit")
if not _logger.handlers:
    _h = logging.FileHandler(_path)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)
_logger.propagate = False

def audit(**kw) -> None:
    _logger.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}, default=str))