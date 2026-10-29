"""Publish selected HF weights only after a fresh-process load/forward succeeds."""

import os
from pathlib import Path
import shutil
from . import DOMAINS
from .common import below, digest, model_inventory, need, now, read, write
from .selection import validate_best


def validate_export(record):
    need(
        record["status"] == "PASS" and record["load_proof"]["status"] == "PASS",
        "Export load proof missing",
    )
    proof = record["load_proof"]
    need(
        proof.get("all_weights_loaded")
        and proof.get("finite_forward")
        and proof.get("selected_identity") == record["selected"]["weights_identity"],
        "Incomplete export load proof",
    )
    need(
        model_inventory(record["path"]) == record["files"], "Exported HF model changed"
    )
    need(
        record["weights_identity"] == digest(record["files"]), "Export identity changed"
    )
    return record


def retire_native_best(run, export):
    validate_export(export)
    selected = export["selected"]
    if selected["step"] and Path(selected["path"]).exists():
        need(
            read(Path(run) / "best.json") == selected,
            "Best changed before native retirement",
        )
        shutil.rmtree(below(selected["path"], Path(run) / "best_weights"))


def finish_export(request):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from multidomain.template import tokenizer_fingerprint

    run, folder = Path(request["run_dir"]), Path(request["job_dir"])
    selected = validate_best(read(run / "best.json"))
    if (run / "export.json").exists():
        previous = validate_export(read(run / "export.json"))
        need(
            previous["selected"] == selected == request["selected"],
            "Export selection changed",
        )
        return previous
    candidate = read(folder / "export_candidate.json")
    need(
        candidate["selected"] == selected == request["selected"],
        "Exported another selection",
    )
    source = Path(candidate["path"])
    if selected["step"]:
        for rank in range(8):
            proof = read(folder / "train" / f"export_rank_{rank}.json")
            need(
                proof["status"] == "PASS"
                and proof["selected_identity"] == selected["weights_identity"]
                and proof["step"] == selected["step"]
                and proof["optimizer_updates"] == 0
                and proof["pid"] != os.getpid(),
                "Missing separate-process native export proof",
            )
    need(
        (source / "config.json").is_file() and list(source.glob("*.safetensors")),
        "HF export incomplete",
    )
    if (source / "model.safetensors.index.json").exists():
        names = set(
            read(source / "model.safetensors.index.json")["weight_map"].values()
        )
        need(
            all(
                (source / n).is_file() and (source / n).stat().st_size > 0
                for n in names
            ),
            "HF shards missing",
        )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    original = AutoTokenizer.from_pretrained(
        request["base_model"], local_files_only=True
    )
    need(
        tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(original),
        "Export lost tokenizer/chat-template identity",
    )
    model, info = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_loading_info=True,
    )
    need(
        not any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ),
        "Fresh HF model did not load all weights",
    )
    model.eval()
    tokens = tokenizer("Compute 1 + 1.", return_tensors="pt").to(
        model.get_input_embeddings().weight.device
    )
    with torch.inference_mode():
        logits = model(**tokens).logits
    need(
        bool(torch.isfinite(logits).all()),
        "Fresh HF forward produced non-finite logits",
    )
    proof = dict(
        status="PASS",
        pid=os.getpid(),
        at_utc=now(),
        all_weights_loaded=True,
        finite_forward=True,
        selected_identity=selected["weights_identity"],
        loading_info=info,
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
    )
    del model, logits
    torch.cuda.empty_cache()
    return publish_export(request, selected, source, proof)


def publish_export(request, selected, source, proof):
    """Recover the rename-before-pointer window using a durable load-proof intent."""
    run, source = Path(request["run_dir"]), Path(source)
    if selected["step"]:
        target = run / "artifacts" / ("hf_step_" + str(selected["step"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        intent_path = run / "export_intent.json"
        if target.exists():
            intent = read(intent_path)
            need(
                intent["selected"] == selected
                and intent["path"] == str(target)
                and model_inventory(target) == intent["files"],
                "Conflicting HF export publication",
            )
            proof = intent["load_proof"]
        else:
            write(
                intent_path,
                dict(
                    selected=selected,
                    path=str(target),
                    files=model_inventory(source),
                    load_proof=proof,
                ),
            )
            source.rename(target)
    else:
        target = source  # Base won validation; preserve its exact immutable HF asset.
    inv = model_inventory(target)
    result = dict(
        status="PASS",
        path=str(target),
        format="huggingface",
        step=selected["step"],
        selected=selected,
        weights_identity=digest(inv),
        files=inv,
        load_proof=proof,
        verifier_identity=request["verifier_identity"],
        lineage=request["lineage"],
    )
    validate_export(result)
    write(run / "export.json", result)
    retire_native_best(run, result)
    return result


def evaluation_model(campaign, domain):
    if domain == "base":
        return dict(
            path=campaign["contract"]["base_model"],
            weights_identity=campaign["contract"]["base_identity"],
            step=0,
        )
    export = validate_export(
        read(Path(campaign["path"]) / "runs" / domain / "export.json")
    )
    return {k: export[k] for k in ("path", "weights_identity", "step")}


def manifest(campaign):
    from .evaluation import read_receipt

    root = Path(campaign["path"])
    teachers, matrix, test_verifiers = {}, {}, {}
    for domain in DOMAINS:
        run = root / "runs" / domain
        if not (run / "export.json").exists():
            return None
        exp = validate_export(read(run / "export.json"))
        teachers[domain] = dict(
            path=exp["path"],
            best_step=exp["step"],
            selection_score=exp["selected"]["score"],
            selection_split="validation",
            weights_identity=exp["weights_identity"],
            verifier_identity=exp["verifier_identity"],
            load_proof=exp["load_proof"],
            lineage=exp["lineage"],
        )
        if (run / "test.json").exists():
            receipt = read_receipt(run / "test.json")
            need(
                receipt["binding"]["weights"] == exp["weights_identity"]
                and receipt["split"] == "test",
                "Test used different weights",
            )
            matrix[domain] = {d: receipt["scores"][d] for d in DOMAINS}
            test_verifiers[domain] = receipt["binding"]["verifier"]
    base = root / "runs/base/test.json"
    base_receipt = read_receipt(base) if base.exists() else None
    if base_receipt:
        test_verifiers["base"] = base_receipt["binding"]["verifier"]
    from .math_test_repair import active_revision

    revision = active_revision(campaign)
    expected_verifier = (
        revision["identity"] if revision else campaign["contract"]["verifier_identity"]
    )
    complete = len(matrix) == 4 and base.exists()
    current = all(v == expected_verifier for v in test_verifiers.values())
    result = dict(
        version="domain-teachers-manifest-v1",
        campaign_identity=campaign["identity"],
        teachers=teachers,
        test_matrix=matrix,
        test_columns=list(DOMAINS),
        base_test=base_receipt["scores"] if base_receipt else None,
        tests_complete=complete,
        test_verifier_identities=test_verifiers,
        expected_test_verifier=expected_verifier,
        tests_use_current_verifier=current,
        comparable_tests_complete=complete and current,
        at_utc=now(),
    )
    write(root / "teachers.json", result)
    return result
