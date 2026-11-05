import json

from api.schemas import EXAMPLE
from incident_risk.config import ROOT

out = ROOT / "examples"
out.mkdir(exist_ok=True)


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2))


batch = lambda n: {
    "events": [
        {
            **EXAMPLE,
            "event_id": f"EX-{i:04d}",
            "error_rate_pct": round((i * 0.97) % 100, 2),
            "queue_depth": (i * 7) % 300,
        }
        for i in range(n)
    ]
}
w("single.json", EXAMPLE)
w("batch_3.json", batch(3))
w("batch_100.json", batch(100))
w("invalid_cpu.json", {**EXAMPLE, "cpu_utilization_pct": 150})
w("unknown_category.json", {**EXAMPLE, "service_name": "brand_new_service"})
(out / "malformed.json").write_text('{"service_name": "inventory",')
print("wrote", sorted(p.name for p in out.iterdir()))
