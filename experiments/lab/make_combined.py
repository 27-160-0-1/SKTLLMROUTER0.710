# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Build data/combined/{inputs,outcomes}.json = train + dev in protocol format."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "data/combined"; out.mkdir(parents=True, exist_ok=True)
for kind, srcs in (("inputs", ["data/materialized/train/inputs.json", "data/materialized/dev/inputs.json"]),
                   ("outcomes", ["data/train/outcomes.json", "data/dev/outcomes.json"])):
    eps, cid, ver = [], None, None
    for s in srcs:
        d = json.loads((ROOT / s).read_text(encoding="utf-8"))
        cid = d["challenge_id"]; ver = d["schema_version"]
        eps.extend(d["episodes"])
    doc = {"schema_version": ver, "challenge_id": cid, "split": "public-train-dev", "episodes": eps}
    (out / f"{kind}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(kind, len(eps))
