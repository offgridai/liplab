import json
from pathlib import Path
from datetime import datetime, timezone

reviewer = "Matthew"
now = datetime.now(timezone.utc).isoformat()

for path in Path("outputs/gold_drafts").glob("*/draft.annotation.json"):
    p = json.loads(path.read_text(encoding="utf-8"))

    p.setdefault("approval", {})
    p["approval"].update({
        "status": "approved_gold",
        "reviewed": True,
        "reviewer": reviewer,
        "reviewed_utc": now,
    })

    layers = p.setdefault("review_layers", {})
    layers.setdefault("speech_regions", {}).update({
        "status": "reviewed_boundary",
        "reviewed": True,
        "reviewer": reviewer,
    })
    layers.setdefault("word_heads", {}).update({
        "status": "reviewed_boundary",
        "reviewed": True,
        "reviewer": reviewer,
    })
    layers.setdefault("dense_visemes", {}).update({
        "status": "reviewed_dense",
        "reviewed": True,
        "reviewer": reviewer,
    })

    path.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    print("approved", path.parent.name)