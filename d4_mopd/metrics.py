"""Report Math sources separately; no evaluation metric changes the training loss."""
from collections import defaultdict
import math
from d4_teachers.common import need
from . import DOMAINS

SELECTION_KEY = "val-selection/four_domain_macro"


def source_metrics(extra):
    keys = ("reward", "mopd_source", "mopd_domain", "mopd_sample_id")
    need(all(k in extra for k in keys), "Validation provenance is incomplete")
    need(len({len(extra[k]) for k in keys}) == 1, "Validation provenance length mismatch")
    need(len(set(extra["mopd_sample_id"])) == len(extra["mopd_sample_id"]),
         "R1 validation contains duplicate prompts")
    values = defaultdict(list)
    domains = defaultdict(list)
    for score, source, domain in zip(extra["reward"], extra["mopd_source"], extra["mopd_domain"]):
        need(math.isfinite(score), "Nonfinite validation reward")
        need(domain in DOMAINS, "Unknown validation domain")
        domains[domain].append(score)
        if domain == "math":
            family = "deepmath" if source in ("deepmath_l8", "deepmath_l9") else "nemotron"
            values["math/" + family].append(score)
            if family == "deepmath":
                values["math/" + source].append(score)
    out = {}
    for key, scores in values.items():
        out["val-source/" + key + "/mean"] = sum(scores) / len(scores)
        out["val-source/" + key + "/n"] = len(scores)
    if all("math/" + k in values for k in ("nemotron", "deepmath")):
        out["val-source/math/equal_source_mean"] = sum(
            out["val-source/math/" + k + "/mean"] for k in ("nemotron", "deepmath")) / 2
        if all(d in domains for d in DOMAINS):
            scores = {d: sum(domains[d]) / len(domains[d]) for d in DOMAINS}
            scores["math"] = out["val-source/math/equal_source_mean"]
            out.update({"val-selection/" + d: v for d, v in scores.items()})
            out[SELECTION_KEY] = sum(scores.values()) / len(DOMAINS)
    return out


def select_best(run, identity, arm):
    """Use only canonical full-validation receipts; ties keep the earlier step."""
    from pathlib import Path
    from d4_teachers.common import read, write, digest
    candidates = []
    for path in sorted((Path(run) / "validation").glob("step_*.json")):
        row = read(path)
        need(row["status"] == "PASS" and row["identity"] == identity and row["arm"] == arm
             and row["split"] == "validation", "Invalid selection receipt")
        score = row["metrics"][SELECTION_KEY]
        need(math.isfinite(score), "Nonfinite selection score")
        candidates.append(dict(step=row["step"], score=score, checkpoint=row["checkpoint"],
                               checkpoint_identity=row["checkpoint_identity"],
                               validation_receipt=str(path), validation_sha256=digest(row)))
    need(candidates, "No full validation result for checkpoint selection")
    selected = max(candidates, key=lambda r: (r["score"], -r["step"]))
    result = dict(selected, identity=identity, arm=arm, selection_key=SELECTION_KEY,
                  split="validation", test_used_for_selection=False)
    write(Path(run) / "best_validation.json", result)
    return result
