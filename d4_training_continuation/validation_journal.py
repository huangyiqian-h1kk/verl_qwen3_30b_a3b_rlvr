"""Durable completed validation batches around the pinned upstream validator.

Only two statements are inserted into the existing loop. Generation, reward
extraction, output logging and metric aggregation remain upstream code. A batch
is reusable only for identical checkpoint, protocol, ordered inputs and runtime.
"""
import ast
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import types
from .common import digest, need, now, read, write

POLICY = dict(version="2026-10-23.validation-batches.1", rows=1006,
              resume="completed-batches-before-next-update", val_n=1,
              validation_shuffle=False, do_sample=False)
VALIDATOR_AST = "e66cab3cf3b01b83e58053a01133f58228635cc35ffbcec79d9d1997ce9afe45"
LISTS = ("sample_inputs", "sample_outputs", "sample_gts", "sample_scores",
         "sample_turns", "sample_uids", "data_source_lst")


def check_source(path):
    tree = ast.parse(Path(path).read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_validate"]
    need(len(found) == 1 and hashlib.sha256(ast.dump(found[0], include_attributes=False).encode()).hexdigest()
         == VALIDATOR_AST, "Installed validation loop differs from pinned verl source")


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    need(value is None or isinstance(value, (str, bool, int, float)),
         "Unsupported validation journal value: " + type(value).__name__)
    need(not isinstance(value, float) or math.isfinite(value), "Non-finite validation journal value")
    return value


def instrument(bound):
    function = bound.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    checksum = hashlib.sha256(ast.dump(definition, include_attributes=False).encode()).hexdigest()
    need(checksum == VALIDATOR_AST, "Installed validation loop differs from pinned verl source")
    loops = [n for n in definition.body if isinstance(n, ast.For)]
    need(len(loops) == 2 and ast.unparse(loops[0].iter) == "self.val_dataloader",
         "Unexpected validation loop structure")
    loop = loops[0]
    loop.body[:0] = ast.parse("if self.cont_journal.begin(test_data, locals()):\n    continue\n").body
    loop.body += ast.parse("self.cont_journal.commit(locals())\n").body
    # No import or global monkey patch; this one bound function gets the hooks.
    definition.decorator_list = []
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), function.__code__.co_filename, "exec"), namespace)
    return types.MethodType(namespace[definition.name], bound.__self__)


def dependency_versions():
    from importlib.metadata import version
    return {p: version(p) for p in ("torch", "vllm", "transformers", "sympy",
                                    "math-verify", "latex2sympy2_extended")}


def runtime_contract(trainer):
    cfg = trainer.config
    need(cfg.data.get("validation_shuffle", True) is False, "Resumable validation requires ordered inputs")
    kw = cfg.actor_rollout_ref.rollout.val_kwargs
    need(kw.n == 1 and kw.do_sample is False, "Resumable validation requires original greedy n=1 decoding")
    need(len(trainer.val_dataset) == POLICY["rows"], "Validation dataset row count changed")
    from omegaconf import OmegaConf
    # Bind actual decoding/tokenization/reward settings, ignoring placement and
    # diagnostic output paths. This catches changes beyond the submit config.
    values = dict(validation=OmegaConf.to_container(kw, resolve=True),
                  data={k: cfg.data.get(k) for k in (
                      "val_batch_size", "max_prompt_length", "max_response_length", "truncation")},
                  tokenizer=str(cfg.actor_rollout_ref.model.path), rows=len(trainer.val_dataset),
                  dataset_sha256=digest(plain(trainer.val_dataset.rows)),
                  dependencies=dependency_versions(),
                  batches=len(trainer.val_dataloader))
    return plain(values)


class Journal:
    def __init__(self, directory, binding, runtime):
        self.directory = Path(directory)
        self.identity = dict(version=POLICY["version"], **binding, runtime=runtime)
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            need(read(manifest) == self.identity, "Validation journal binding changed")
        else:
            write(manifest, self.identity)
        self.index = 0
        self.rows = 0
        self.cached = 0
        self.offsets = None

    def begin(self, data, scope):
        self.index += 1
        self.input_digest = digest(plain(data))
        path = self.directory / (f"batch_{self.index:06d}.json")
        self.offsets = {k: len(scope[k]) for k in LISTS}
        self.extra_offsets = {k: len(v) for k, v in scope["reward_extra_infos_dict"].items()}
        if not path.exists():
            return False
        envelope = read(path)
        payload = envelope["payload"]
        need(envelope["sha256"] == digest(payload) and payload["batch"] == self.index
             and payload["input_digest"] == self.input_digest, "Cached validation batch changed or inputs reordered")
        self.check(payload)
        for k in LISTS:
            scope[k].extend(payload["lists"][k])
        for k, values in payload["extras"].items():
            scope["reward_extra_infos_dict"].setdefault(k, []).extend(values)
        self.rows += payload["rows"]
        self.cached += 1
        self.progress()
        return True

    def check(self, payload):
        n = payload["rows"]
        need(n > 0 and all(len(payload["lists"][k]) == n for k in (
            "sample_inputs", "sample_outputs", "sample_gts", "sample_scores", "sample_uids")),
            "Incomplete validation batch")
        need(all(len(v) in (0, n) for v in payload["extras"].values()), "Incomplete batch reward details")
        need(sum(len(v) for v in payload["lists"]["data_source_lst"]) == n, "Missing batch domain labels")
        need(all(isinstance(x, str) for k in ("sample_inputs", "sample_outputs", "sample_uids")
                 for x in payload["lists"][k]), "Invalid validation strings")
        plain(payload)

    def commit(self, scope):
        need(self.offsets is not None, "Validation batch was not started")
        values = {k: plain(scope[k][self.offsets[k]:]) for k in LISTS}
        payload = dict(batch=self.index, input_digest=self.input_digest, rows=len(values["sample_scores"]),
                       lists=values, extras={k: plain(v[self.extra_offsets.get(k, 0):])
                           for k, v in scope["reward_extra_infos_dict"].items()})
        self.check(payload)
        write(self.directory / f"batch_{self.index:06d}.json", dict(payload=payload, sha256=digest(payload)))
        self.rows += payload["rows"]
        self.progress()

    def progress(self):
        write(self.directory / "progress.json", dict(completed_batches=self.index, completed_rows=self.rows,
              cached_batches_this_process=self.cached, expected_rows=POLICY["rows"], at_utc=now()))

    def finish(self):
        need(self.rows == POLICY["rows"] and self.index == self.identity["runtime"]["batches"],
             "Validation journal is incomplete")
        uids = []
        for index in range(1, self.index + 1):
            envelope = read(self.directory / f"batch_{index:06d}.json")
            need(envelope["sha256"] == digest(envelope["payload"]), "Validation batch checksum changed")
            uids.extend(envelope["payload"]["lists"]["sample_uids"])
        need(len(set(uids)) == self.rows, "Duplicate validation prompt IDs")
        return dict(path=str(self.directory), rows=self.rows, batches=self.index,
                    reused_batches=self.cached, manifest_sha256=digest(self.identity))
