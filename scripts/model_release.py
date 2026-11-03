import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS, ARCHIVE = ROOT / "models", ROOT / "models" / "archive"
FILES = ["model.joblib", "preprocessor.joblib", "artifact_manifest.json", "metadata.json"]

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def archive() -> None:
    m = json.loads((MODELS / "artifact_manifest.json").read_text())
    if m["model_sha256"] != sha(MODELS / "model.joblib"):
        sys.exit("Live model.joblib does not match its manifest. Run `make register` first.")
    dest = ARCHIVE / m["version"]
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (MODELS / f).exists():
            shutil.copy2(MODELS / f, dest / f)
    print("archived", dest)

def list_versions() -> None:
    for d in sorted(ARCHIVE.glob("*/artifact_manifest.json")):
        m = json.loads(d.read_text())
        print(f"{m['version']:24s} model={m['model']:20s} threshold={m['threshold']:.4f}")

def rollback(version: str) -> None:
    src = ARCHIVE / version
    if not (src / "artifact_manifest.json").exists():
        sys.exit(f"No archived version {version!r}. Run `list`.")
    m = json.loads((src / "artifact_manifest.json").read_text())
    if m["model_sha256"] != sha(src / "model.joblib"):
        sys.exit("Archived model fails its checksum. Refusing to restore.")
    try:
        archive()                      # keep a copy of what is live right now
    except SystemExit as e:
        print("warning: live model not archived:", e)
    for f in FILES:
        if (src / f).exists():
            shutil.copy2(src / f, MODELS / f)
    print(f"restored {version}. Now restart the API: docker compose restart api")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["archive", "list", "rollback"])
    ap.add_argument("version", nargs="?")
    a = ap.parse_args()
    {"archive": archive, "list": list_versions}.get(a.cmd, lambda: rollback(a.version))()