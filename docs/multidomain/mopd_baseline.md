# 四域 plain MOPD：实现、验收与运行

这份实现用于 mix rl 的基础实验：固定 Math50、IF70、CP80、Logic30，比较 **rollout=1 的 domain-mix 与 domain-only optimizer batch**。训练只有蒸馏信号。原 teacher campaign 作为只读输入，新实验、环境、日志和 checkpoint 使用独立目录。

**验收状态：本地 CPU 测试已通过；ABCI GPU 资格测试尚未执行。** 本地没有 ABCI 模型文件、PBS 或 GPU，不能据此宣称 Qwen3-30B-A3B 的分布式训练已经跑通。正式训练入口会检查该 arm 的 GPU 资格测试收据。

## 1. 依据与固定版本

- 原始 [MOPD 论文](https://arxiv.org/html/2606.30406v1)：采用 §3.2.1 的 sampled-token PG 形式，response 内 token 平均、batch 内 response 平均；附录给出 BS=2048、N=1、关闭 Dynamic Sampling、PG signal clip=±5。这里选择 PG 版本，不把 top-k 版本叠加进来。
- 固定 verl commit：`482f86fd09959912c99c0acf2b90329f9e061da7`。
- 复用 [官方 Megatron OPD 示例](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/on_policy_distillation_trainer/run_qwen3_8b_megatron.sh)、[官方多 teacher 示例](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh) 与 [Qwen3-30B-A3B Megatron 配置](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh)。这些接口的组合需要本项目的 GPU 资格测试；上游并没有一个可以直接照搬的四个 30B MoE teacher 完整示例。
- 不复制 trainer、loss 或 optimizer 的实现；使用上游 `RayPPOTrainer`、`MultiTeacherModelManager`、`distillation_loss`、`compute_policy_loss_gpg` 与 Megatron model engine。项目适配层负责数据、检查、完整 checkpoint 收据和 PBS 控制。

## 2. 已冻结的实验定义

| 项目 | 本次实现 | 来源/性质 |
|---|---|---|
| Student 起点 | 原 teacher campaign 的 `contract.base_model` | 同源起点；没有新做一次 SFT |
| Math teacher | `runs/math_step50/export.json`，step 50 | 用户指定；不是 `math_continued` 的另一份评估记录 |
| IF teacher | `runs/if_step70/export.json`，step 70 | 用户指定；不读取旧的 adopted IF40 指针 |
| CP teacher | `runs/cp_step80/export.json`，step 80 | 用户指定 |
| Logic teacher | `runs/logic_algorithmic/export.json`，step 30 | 用户指定 |
| Prompt batch / rollout | 2048 prompts × 1 response | 对齐论文 MOPD |
| 蒸馏目标 | sampled-token PG，teacher−student logprob，detach，clip±5 | 对齐论文 PG 形式 |
| Loss reduction | `seq-mean-token-mean` | 每条 response 等权，排除 prompt/padding；保留有效 EOS |
| 每批更新 | `ppo_mini_batch_size=2048`，`ppo_epochs=1` | 一批新 rollout → 一次 optimizer update → 丢弃 rollout |
| 优化器 | Megatron Adam/AdamW 方式，LR=1e-6，betas=(0.9,0.999)，WD=.01，grad clip=1 | 上游 OPD/优化器默认值与项目工程选择；论文没有单独明确披露全部 MOPD optimizer 数值 |
| LR schedule | constant，warmup=0 | 明确工程默认值 |
| 域比例 | Math/IF/CP/Logic 各 1/4 | 四域适配；不是论文三域比例的原样复现 |
| 域内采样 | 固定数据集内 shuffle，耗尽后再 shuffle | 不重新配平 Math 内 Nemotron/DeepMath，不按 reward 过滤 |
| Rollout decoding | temperature=1，top_p=1，top_k=-1 | 原始分布采样；单次 assistant response |
| Context | 沿用项目每条记录的 64K 总预算和冻结 chat template | 项目适配；不是论文 Math/IF 的 32K 配置 |
| 正式预算 | 200 updates，每 4 步保存，每 20 步 validation | 工程默认值；固定预算，无 adaptive early stop |
| Validation | 原四域 validation + 冻结 DeepMath validation，greedy n=1 | 沿用 teacher 评估协议；不同于论文数学 avg@32 |
| 额外训练信号 | 无 task reward、reference KL、entropy bonus、SFT、value loss、PPO ratio clipping、importance correction、动态过滤；显式关闭 MoE auxiliary/z loss | 纯蒸馏 baseline |

默认每个 arm 共 409,600 prompt exposures，每域 102,400。重复采样不代表新增独立样本。`seed=42` 固定数据计划与模型随机种子；不同 batching 下的生成结果不要求相同。

论文没有披露的工程参数在表中单独标出，不声称逐 bit 复现原论文。改变 B、总步数、保存或验证频率时，使用新的 spec 文件和输出目录；这些计数必须为 4 的倍数。

## 3. Loss 与更新语义

对 student 自己生成的 response，令有效 response token 数为 `L_i`，被该 prompt 路由到的 teacher 为 `T(d_i)`：

```text
a[i,t] = stop_gradient(clip(log p_teacher(y[i,t] | prefix)
                           - log p_student(y[i,t] | prefix), -5, +5))
loss = -(1/B) * sum_i ((1/L_i) * sum_valid_t a[i,t] * log p_student(y[i,t] | prefix))
```

Student logprob 在训练 forward 上计算。Teacher 只对同一份 student token prefix 做 prefill scoring，不生成另一份答案、不参与反向传播。使用的 sampled `k1` 项可能为负，不应把单条样本的该值当成非负 full-vocabulary KL。

固定的上游版本已经有 `gpg` loss，但 `DistillationLossConfig` 的构造检查仅允许 `vanilla`。`compat.BasicPGConfig` 只放行本配方的 `k1 + PG + gpg + clip5 + no task reward` 组合；实际 loss/backward 仍调用上游函数。CPU 测试对正负信号、clip、teacher detach、不同长度和 microbatch/DP 归一化做了数值验证。

配置里的 `algorithm.adv_estimator=grpo` 是上游 trainer 的零 task-reward 数据通路；蒸馏优势由上面的 token logprob 差生成。**没有用 R=1 做组内 reward 归一化来产生蒸馏梯度。** `0-advantage group fraction` 不适合作为本次 MOPD 有效样本比例。

## 4. 两种 batching 的公平比较

| 维度 | domain-mix | domain-only |
|---|---|---|
| 每次 optimizer update | 每域 512 prompts，共 2048 | 单一域 2048 prompts |
| 每四次 update | 每域 2048 exposures | 每域恰好一次 update，共 2048 exposures |
| 数据计划 | 使用每域相同的 shuffle stream | 使用同一组 stream |
| 域顺序 | batch 内打乱 | 每四步轮换首域，四轮构成 Latin rotation |
| 梯度累积 | 累积到完整 mixed batch 后更新一次 | 累积到完整单域 batch 后更新一次 |

逐四步比较两 arm 的 prompt **多重集**，包含重复样本次数。Domain-only 的意思是完整 optimizer update 同域，不能通过把四域 microbatch 累积到一起实现。Native token balancing 只改变完整 batch 内的顺序；运行时仍检查域计数和 sample ID 多重集。

建议主比较使用 step=0/20/40/...：这些点刚好完成整组四域预算。中间单域更新后的 seesaw 可另作诊断，不能与另一 arm 暴露不相等的点直接混算。

## 5. 资源与环境

每个 arm 使用 2 个完整 rt_HF 节点，共 16 GPU：

- Student 节点：Megatron TP2/PP2/EP2；student rollout TP4；参数和 optimizer offload。
- Teacher 节点：4 个冻结 teacher，各 1 个 TP2 实例，总共 8 GPU。
- 同时启动两个 arm 需要 4 节点/32 GPU。Domain-only 仍保留相同 teacher 资源预算，即使当前 batch 只用一个 teacher。

该 pinned verl 的 `uv.lock` 使用 **PyTorch 2.13 / CUDA 13 / vLLM 0.32**。不能直接在原 teacher 的 CUDA 12.8 环境里 `pip install -U`。Bootstrap 创建单独环境；GPU preflight 检查扩展导入、BF16 运算和显存。ABCI driver 与该 CUDA 13 栈是否兼容仍需实机验证，检查不通过时不提交正式训练。

默认保留全部完整 checkpoint：两个正式 arm 共 100 份，另有 4 份 qualification checkpoint。CPU preflight 输出可用文件系统空间及按 BF16 权重×7 估算的总 checkpoint 体积；30B 全参数 Adam 状态会占用数十 TB。估算不包含中断留下的半成品，也不能验证项目 quota，需结合 ABCI quota 核对。当前实现不会自动删除历史完整 checkpoint。

Teacher 推理 `max_model_len` 为 padding 上界与 prompt-logprob API 预留一个 token；实际每条 student prompt+response 仍受冻结的 65,536 token 总预算限制。

## 6. 三层验收

| 层级 | 检查内容 | 当前证据 |
|---|---|---|
| 本地 CPU 软件测试 | 官方 Hydra/dataclass、原生 loss 与梯度、原生 teacher routing、数据 factory、token 对齐、采样/loader 恢复、checkpoint 与 PBS 状态机 | `tests/mopd` 已通过，见下面复跑命令 |
| ABCI CPU preflight | 实际 base/teacher 导出身份、数据分割、prompt token hash、tokenizer、冻结 verifier、依赖与配置 | 必须在 ABCI 执行，尚未执行 |
| ABCI GPU qualification | 真实四 teacher scoring、Megatron forward/backward/update、原生完整 checkpoint；step4 正常退出，再由新 PBS 进程恢复到 step8 | 必须分别完成两个 arm，尚未执行 |

Qualification 使用 **B=8、8 updates、response 上限256**，从每域最长的两个训练 prompts 取测试样本；validation 每域一个样本。路径位于 `qualification/`，不改变正式配方。它验证基本执行与恢复，**不证明 B2048/64K 的最坏长度吞吐和显存容量**。正式运行出现 OOM 会停在 BLOCKED，不会改 batch 或截断长度来伪装通过。

CPU 中的 AdamW 小模型恢复测试检查保存状态后下一次更新完全一致；它不替代上述真实 MoE 重启测试。分布式 loss 的 CPU 测试模拟 DP 梯度平均，实际 NCCL/DP 通信留给 GPU qualification。

## 7. ABCI 操作命令

在项目代码包含本分支的 checkout 中运行，任一登录节点均可。推荐为 MOPD 建独立 worktree，避免改变已有 teacher checkout。代码通过 Git 管理，不需要下载或应用 patch 文件。

首次创建 MOPD checkout（以下获取命令以 `codex/d4-mopd-baseline` 已推送为前提；仅有本地提交时不能在 ABCI fetch 到它）：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_teachers
git fetch --no-tags origin codex/d4-mopd-baseline
git worktree add --detach ../data_mixture_rl_4d_mopd FETCH_HEAD
)
```

这条 `worktree add` 仅首次执行；如果目标目录已存在，先核对该目录的版本和未提交修改，再决定是否更新，不能覆盖正在使用的 checkout。

先设置目录并创建新环境（需要可用的 `uv`）：

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd

mopd_verl=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd
bash scripts/mopd_bootstrap.sh "$mopd_verl"
```

准备数据与 manifest，再做 CPU preflight。这一步只读取 teacher campaign，不提交 GPU 作业：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_verl=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd
mopd_python="$mopd_verl/.venv/bin/python"
mopd_teacher=/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_teachers/outputs/domain_teachers/qwen30b_d4_teachers_seed42_v1
mopd_output="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_v1"

"$mopd_python" -I scripts/mopd.py prepare \
  --teacher-campaign "$mopd_teacher" --output "$mopd_output" \
  --verl-src "$mopd_verl" --python "$mopd_python"
"$mopd_python" -I scripts/mopd.py preflight --manifest "$mopd_output/manifest.json"
)
```

提交 GPU qualification 并启动监护：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_v1/manifest.json"
"$mopd_python" -I scripts/mopd.py start --manifest "$mopd_manifest" --phase qualification --arm both
"$mopd_python" -I scripts/mopd.py status --manifest "$mopd_manifest"
timeout 20s qstat -u "$USER" || true
)
```

两个 `qualification/*/receipt.json` 均 PASS 后，显式启动正式训练。命令本身也会检查收据，不会因为只有 CPU PASS 就开始训练：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_v1/manifest.json"
"$mopd_python" -I scripts/mopd.py start --manifest "$mopd_manifest" --phase train --arm both
)
```

资源不足时可用 `--arm domain_mix` 或 `--arm domain_only` 分别启动。后面的简短状态/暂停命令需要在登录 shell 中重新设置 `mopd_python`、`mopd_manifest` 并进入 checkout；上面的 subshell 不会保留这些变量。

## 8. Checkpoint、监护和中断

- 原生 checkpoint 必须包含 model、optimizer、LR scheduler、worker RNG，再加 DataLoader state 和 driver RNG。所有文件完成后才原子写入 `mopd_complete.json`。
- 重启只使用最后一个完整 marker；半写入的新 checkpoint 移入当前 attempt 的 `incomplete_checkpoints/` 保留证据。已提交 checkpoint 被修改会报错，避免悄悄回退。
- 两 arm 独立保存，marker 绑定 manifest、arm、step、sampler 身份。不会把 Math50 权重当作 student resume，也不会只加载权重冒充续训。
- 未保存的 rollout/update 会在最后一次完整 checkpoint 后重做。**vLLM 的采样 RNG 没有逐请求完整恢复保证**，因此重做 response 不承诺 bitwise 一致。
- PBS 271、严格识别的 root-terminated265、已识别 walltime 中断可以重新排队；明确的应用 FAIL（如 OOM、对齐错误）保持 BLOCKED。qstat timeout/unknown 保留原 job；不创建第二个相同任务。
- qsub 前持久化 pending attempt，模糊的提交返回通过同一目录恢复。任务身份检查包含 owner、queue、job name 与脚本路径。
- Guardian 使用共享文件系统锁，不要求回到原 login host，也不 SSH 到另一个 login 节点发送 PID signal。重启入口可在任一登录节点运行；已有 guardian 时复用。全部任务终态时退出。
- 登录节点重启可能使 guardian 消失；它不能在没有任何存活进程的情况下自己复活。已有 PBS 作业独立继续，下次 `resume-control` 接管同一任务记录。

```bash
# 只读状态
"$mopd_python" -I scripts/mopd.py status --manifest "$mopd_manifest"
# 停止后续提交，运行中的训练在下一 update 边界暂停；不 qdel
"$mopd_python" -I scripts/mopd.py stop --manifest "$mopd_manifest"
# 清除显式 STOP、启动/复用监护；应用错误 BLOCKED 不自动清除
"$mopd_python" -I scripts/mopd.py resume-control --manifest "$mopd_manifest"
```

## 9. 查看结果与测量时间

`status` 返回 guardian 锁、heartbeat、每个任务及上次 PBS 观察。每个 `runs/<arm>/attempts/<id>/` 保留：

| 文件 | 含义 |
|---|---|
| `request.json` / `resolved.yaml` | 绑定任务与最终展开的配置 |
| `gpu_node_0.json` / `gpu_node_1.json` | 每节点导入/BF16 检查 |
| `resume.json` | 从哪个完整 checkpoint 恢复 |
| `metrics.jsonl` | 上游完整逐步 metrics，包括 generation/update/step timing |
| `update_<step>.json` | 域计数、sample ID 摘要、teacher−student gap、clip 比例、长度、actor update 耗时 |
| `validation_<step>.json` | 本次已完成 validation 的 metrics |
| `validation_generations/<step>.jsonl` | validation responses、score、sample ID 与 source |
| `result.json` / `pbs.log` | 本次进程结果与完整日志 |

Math validation 同时报告 Nemotron、DeepMath、L8/L9 与两个来源等权均值；原生 Math 聚合值仍是按样本数加权，两者名称不同。Task verifier 只用于 validation，通过原环境的隔离 Python 子进程运行；训练不会调用它。

比较收敛时间时分开统计 optimizer update、generation、完整 step、validation/checkpoint 开销、队列等待及中断重做。不要把 `actor_update_seconds` 当成完整训练步耗时，也不要把重跑的同一步累计为新增训练步。中断 attempt 的 metrics 保留，不覆盖。

该阶段冻结了 test 集并检查其不进入训练，但**不在训练中自动反复跑 test，也未实现最终 test 任务调度**；目前完整实现的评估通路是周期 validation。后续最终 checkpoint 的 test 应沿用冻结 verifier 和独立收据，不用 test 结果调整这次训练的批次机制。

## 10. 复跑本地 CPU 测试

CPU 测试环境与 GPU 环境分离。完整 CPU 依赖清单在 `tests/mopd/requirements-cpu.lock.txt`；GPU 环境以 pinned verl 的 `uv.lock` 为准，不能拿 CPU 清单替换它。

```bash
uv venv --python 3.12 .venv-mopd-cpu-test
uv pip install --python .venv-mopd-cpu-test/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  -r tests/mopd/requirements-cpu.lock.txt
D4_MOPD_VERL_SRC=/absolute/path/to/pinned/verl \
  .venv-mopd-cpu-test/bin/python -m pytest tests/mopd -q
python3 -m compileall -q d4_mopd scripts/mopd.py tests/mopd
bash -n d4_mopd/node.sh scripts/mopd_bootstrap.sh
```

本次本地结果：**45 passed**。这些测试直接导入固定版本的上游 loss/config/teacher manager；缺少 upstream checkout 或导入失败会失败，不会用 skip 冒充通过。

## 11. 文件导览

`config.py` 与 `compat.py` 接上游配方；`assets.py` 冻结真实资产；`spec.py` 定义两 arm 的数据计划；`integration.py` 在原生 trainer 边界检查并提交 checkpoint；`checkpoints.py` 负责完整性；`preflight.py` 分开 CPU/GPU 证据；`control.py`、`node.sh`、`cli.py` 负责 PBS；`metrics.py` 与 `reward.py` 保留来源级 validation。所有修改均在新增 `d4_mopd` 等文件内，未改原 teacher 控制器或上游 verl 源码。
