"""Report Math sources separately; no evaluation metric changes the training loss."""
from collections import defaultdict
import math
from d4_teachers.common import need


def source_metrics(extra):
    keys = ("reward", "mopd_source", "mopd_domain", "mopd_sample_id")
    need(all(k in extra for k in keys), "Validation provenance is incomplete")
    need(len({len(extra[k]) for k in keys}) == 1, "Validation provenance length mismatch")
    need(len(set(extra["mopd_sample_id"])) == len(extra["mopd_sample_id"]),
         "R1 validation contains duplicate prompts")
    values = defaultdict(list)
    for score, source, domain in zip(extra["reward"], extra["mopd_source"], extra["mopd_domain"]):
        need(math.isfinite(score), "Nonfinite validation reward")
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
    return out
