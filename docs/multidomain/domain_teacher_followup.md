# Continuing Math, evaluating IF70, and adopting CP80

This opt-in extension adds two tasks to an existing domain-teacher campaign:

| Task | Work |
| --- | --- |
| `if_step70` | Export the final IF70 checkpoint and evaluate the original four-domain test, including Nemotron Math. |
| `math_continued` | Restore the complete Math50 checkpoint, continue training with two-source validation, export the selected checkpoint, then evaluate it. |

The existing CP80 export and completed four-domain test are reused. CP80 is recorded as an explicit user choice after supplementary evaluation, while the original validation-selected CP record remains available. IF70 is an additional measurement and does not automatically replace the original IF teacher.

## Run

Update the existing checkout to the follow-up implementation. Set the following variables to the compatible Python executable and existing campaign directory:

```bash
teacher_python=/path/to/compatible/python
teacher_campaign=/path/to/campaign

"$teacher_python" -I scripts/continue_domain_teachers.py start --campaign "$teacher_campaign"
"$teacher_python" -I scripts/continue_domain_teachers.py status --campaign "$teacher_campaign"
"$teacher_python" -I scripts/inspect_domain_teachers.py --campaign "$teacher_campaign" --logs 8
```

`start` verifies the required checkpoints, sampling metadata, prepared DeepMath held-outs, and completed CP80 artifacts before handing off the guardian. Registration is atomic and repeatable: existing job ownership and progress are preserved. The returned guardian state identifies submitted jobs. Preparing or updating source code alone does not submit jobs.

## Math continuation

- Restore model, optimizer, LR scheduler, RNG, and dataloader cursor from full step50 into a separate run.
- Preserve the original training data mixture, 200-step sampling stream, and LR schedule so the checkpoint cursor remains compatible. A separate execution limit stops the new run by global step100.
- Re-evaluate step50 on both validation sources before updating weights. Evaluate every10 steps and save every5 steps.
- Select by `0.5 * Nemotron_validation_mean + 0.5 * DeepMath_validation_mean`. Report both source scores separately. Neither four-domain averages nor test scores select the continued Math teacher.
- Apply early stopping no earlier than step80, with patience25 steps. Ties retain the earlier checkpoint. The selected checkpoint may remain step50.
- Final evaluation combines the original four-domain test with the prepared DeepMath test. DeepMath test rows do not enter training or validation.

New tasks use the campaign's installed compatible Math verifier, with its identity checked against the prepared DeepMath data. Original training/validation records remain unchanged. Evaluation caches bind the new stage, checkpoint, data, verifier, dependencies, and decoding settings.

## Recovery and selection records

The existing guardian owns submission and recovery. Confirmed PBS271 interruptions and supported walltime expiry resume from the latest complete checkpoint or completed evaluation batches. Unknown scheduler observations retain ownership of the existing job. Repeated interruptions are recorded without introducing a three-retry limit. Existing task and campaign stop controls apply to the new task names.

`teacher_followup.json` freezes the plan. `selected_teachers.json` exposes adopted models even when another teacher is unfinished. CP80 has explicit-choice provenance and no validation-selection score. Math switches to the new stage after its selected export passes validation. The complete `teachers.json` follows the same selection once all teacher exports exist. Its four-domain Math column remains Nemotron-only; additional DeepMath results appear in `math_test_sources`.

CPU tests cover full-state pins, actual torchdata cursor restoration, validation journal recovery across sources, selection and stopping boundaries, interrupted registration, repeated271 recovery, existing-job adoption, duplicate-start prevention, export/test progression, and reporting. GPU model restoration and training require the cluster's runtime checks.
