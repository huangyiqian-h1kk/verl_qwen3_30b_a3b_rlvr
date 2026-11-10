# 四域 plain MOPD：实现、验收与运行

这份实现用于 mix rl 的基础实验：固定 Math50、IF70、CP80、Logic30，比较 **rollout=1 的 domain-mix 与 domain-only optimizer batch**。训练只有蒸馏信号。原 teacher campaign 作为只读输入，新实验、环境、日志和 checkpoint 使用独立目录。

**验收状态：两组 ABCI GPU 资格测试在 Transformer Engine 导入时失败，尚未进入 Ray/训练。** `2893794.pbs1`、`2893931.pbs1` 均为 exit 1，wheelhouse 的 `libtransformer_engine.so` 要求 `GLIBC_2.38`。此前 CPU preflight 只验包版本，未拦住系统 ABI 不兼容；本次补上原生库检查、同版本源码重建及显式重试入口。本地没有 ABCI 模型文件、PBS 或 GPU，不能据此宣称源码构建或 Qwen3-30B-A3B 分布式训练已经跑通。`start --phase pipeline --arm both` 会并行调度两组 GPU 资格测试，全部通过后自动提交两个正式训练任务，随后各自执行最终 test。

## 1. 依据与固定版本

- 原始 [MOPD 论文](https://arxiv.org/html/2606.30406v1)：采用 §3.2.1 的 sampled-token PG 形式，response 内 token 平均、batch 内 response 平均；附录给出 BS=2048、N=1、关闭 Dynamic Sampling、PG signal clip=±5。这里选择 PG 版本，不把 top-k 版本叠加进来。
- 固定 verl commit：`482f86fd09959912c99c0acf2b90329f9e061da7`。
- 复用 [官方 Megatron OPD 示例](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/on_policy_distillation_trainer/run_qwen3_8b_megatron.sh)、[官方多 teacher 示例](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh) 与 [Qwen3-30B-A3B Megatron 配置](https://github.com/verl-project/verl/blob/482f86fd09959912c99c0acf2b90329f9e061da7/examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh)。这些接口的组合需要本项目的 GPU 资格测试；上游并没有一个可以直接照搬的四个 30B MoE teacher 完整示例。
- 不复制 trainer、loss 或 optimizer 的实现；使用上游 `RayPPOTrainer`、`MultiTeacherModelManager`、`distillation_loss`、`compute_policy_loss_gpg` 与 Megatron model engine。项目适配层负责数据、检查、完整 checkpoint 收据和 PBS 控制。

## 2. 已冻结的实验定义

| 项目 | 本次实现 | 来源/性质 |
|---|---|---|
| Student 起点 | 原 teacher campaign 的 `contract.base_model` | 同源起点；没有新做一次 SFT |
| Math teacher | `runs/math_step50/export.json`，step 50 | 用户指定的原训练 checkpoint；资产校验用于防止错载，不要求重新选择 Math50 |
| IF teacher | `runs/if_step70/export.json`，step 70 | 用户指定；不读取旧的 adopted IF40 指针 |
| CP teacher | `runs/cp_step80/export.json`，step 80 | 用户指定 |
| Logic teacher | `runs/logic_algorithmic/export.json`，step 30 | 用户指定 |
| Prompt batch / rollout | 2048 prompts × 1 response | 对齐论文 MOPD |
| 蒸馏目标 | sampled-token PG，teacher−student logprob，detach，clip±5 | 对齐论文 PG 形式 |
| Loss reduction | `seq-mean-token-mean` | 每条 response 等权，排除 prompt/padding；保留有效 EOS |
| 每批更新 | `ppo_mini_batch_size=2048`，`ppo_epochs=1` | 一批新 rollout → 一次 optimizer update → 丢弃 rollout |
| 优化器 | Megatron Adam/AdamW 方式，LR=1e-6，betas=(0.9,0.999)，epsilon=1e-8，WD=.01，grad clip=1 | beta/epsilon 经 `override_optimizer_config` 传入底层构造；这些数值是工程选择，原论文没有完整披露 |
| LR schedule | constant，warmup=0 | 明确工程默认值 |
| 域比例 | mix 每步四域等量；only 每次 update 独立以各 1/4 概率选域 | only 允许连续选择同一域，不强制四步配平；固定 seed，可恢复 |
| 域内采样 | 固定数据集内 shuffle，耗尽后再 shuffle | 不重新配平 Math 内 Nemotron/DeepMath，不按 reward 过滤 |
| Rollout decoding | temperature=1，top_p=1，top_k=-1 | 原始分布采样；单次 assistant response |
| Context | 沿用项目每条记录的 64K 总预算和冻结 chat template | 项目适配；不是论文 Math/IF 的 32K 配置 |
| 正式预算 | 200 updates，每 5 步保存，每 10 步 validation | 用户指定验证间隔；固定预算，无 adaptive early stop |
| Checkpoint 保留 | 每个正式 arm 保留最近 2 个完整 checkpoint，另保留 validation 最佳及已冻结最终 test 的选择 | 保存成功后滚动清理；完整模型/optimizer/scheduler/RNG/数据状态；日志和评估历史保留 |
| Validation | 固定 1,024 题：Math 256（Nemotron 128 + DeepMath 128），IF/CP/Logic 各 256；greedy n=1 | 接近原 Mix RL 的规模；不同于论文数学 avg@32 |
| 选模与最终 test | Math 两来源各 1/2，再四域各 1/4；取完整 validation 分数最高的单一 checkpoint | 平分取较早步；包含 step0；训练结束自动注册独立 test 作业并持续监护 |
| 额外训练信号 | 无 task reward、reference KL、entropy bonus、SFT、value loss、PPO ratio clipping、importance correction、动态过滤；显式关闭 MoE auxiliary/z loss | 纯蒸馏 baseline |

默认每个 arm 共 409,600 prompt exposures。Mix 每域恰好 102,400；only 每域期望 102,400，实际次数由固定随机域计划决定，CPU preflight 和逐步日志记录实际 exposure。重复采样不代表新增独立样本。域选择 RNG 独立于域内 shuffle 和生成 RNG；`seed=42` 固定计划，恢复继续相同域序。

论文没有披露的工程参数在表中单独标出，不声称逐 bit 复现原论文。改变 B、总步数、保存或验证频率时，使用新的 spec 文件和输出目录。B 必须能被四域均分；总步数和频率必须为正整数，验证间隔必须为保存间隔的整数倍，以便每个选模候选都有完整 checkpoint。

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
| 域预算 | 每步四域各 1/4 | 每次独立均匀选域，有限步数内的域次数有随机波动 |
| 数据计划 | 使用每域相同的 shuffle stream | 使用同一组 stream |
| 域顺序 | batch 内打乱 | 专用 seed 产生 IID uniform 随机域序 |
| 梯度累积 | 累积到完整 mixed batch 后更新一次 | 累积到完整单域 batch 后更新一次 |

两 arm 的总 prompt/update 预算相同、各域使用相同 shuffle stream，但 IID only 不保证逐四步或整个实验的各域 exposure 相等。比较时必须同时报告实际域次数。Domain-only 指完整 optimizer update 同域；Native token balancing 只改变完整 batch 内顺序，运行时检查随机计划对应的域及 sample ID 多重集。

正式训练只有一套完整 validation，在 step=0/10/20/.../200 评估，共 21 个计划观测点；step 指 optimizer update。两个 arm 使用同一固定集合，没有每步小 probe。

原 Mix RL 的 validation 是 1,006 题（Nemotron Math 238，其余三域各 256）；上一版 MOPD 直接追加 DeepMath 256 后为 1,262 题。当前从这些既有 validation 数据中固定抽取 Math 两来源各 128 题，其余三域全部保留，总计 1,024 题。DeepMath 依照候选集 L8:L9=213:43 的比例，用最大余数法分配为 L8 107 题、L9 21 题。抽样只依赖 seed=42、来源和 sample ID，prepare 时冻结，不随训练重新抽样，不使用 test 或训练数据。Manifest 保留候选及选中 ID 摘要、来源计数和 Parquet 哈希；CPU preflight 验证四域题数及 Math 的 1:1 比例。已有 manifest 不会被此配置原地改写，下面命令使用新的输出目录。

Validation 的 verifier performance 不参与训练采样、teacher 路由、loss 或 optimizer；只记录结果并选择 checkpoint。当前固定 200 updates，不使用 validation early stopping。`trainer.test_freq` 是上游历史命名，训练期间它调用的是 validation，不读取 test set。

选模分数为 `Math=(Nemotron+DeepMath)/2`，`score=(Math+IF+CP+Logic)/4`。DeepMath 内按固定 L8/L9 样本集合求均值。`runs/<arm>/validation/step_<n>.json` 将完整评估绑定至模型保存记录；`best_validation.json` 指向最高分 checkpoint，分数相同取较早步。中断恢复复用已完成收据，不因重启多次重评并挑高分；非计划评估步不额外选模。

训练完成后，guardian 自动注册 `test:<arm>`，从冻结的 validation 选择加载权重（若 step0 最好则加载 base），通过原生 `val_only` 在 test 数据上评估，绝不调用 actor update。结果在 `final_tests/<arm>/receipt.json`。该作业也处理 271、带管理员终止证据的 265 与 qstat unknown；应用错误阻塞，不用重试掩盖。中断发生在完整 test 收据写出前时重做该次 test；已完成整套收据可复用，尚不缓存未完成的 test 批次。固定预算终点的 validation 另行保留，不从 test 分数反向选择模型。

## 5. 资源与环境

每个 arm 使用 2 个完整 rt_HF 节点，共 16 GPU：

- Student 节点：Megatron TP2/PP2/EP2；student rollout TP4；参数和 optimizer offload。
- Teacher 节点：4 个冻结 teacher，各 1 个 TP2 实例，总共 8 GPU。
- 同时启动两个 arm 需要 4 节点/32 GPU。Domain-only 仍保留相同 teacher 资源预算，即使当前 batch 只用一个 teacher。

该 pinned verl 的 `uv.lock` 使用 **PyTorch 2.13 / CUDA 13 / vLLM 0.32**。不能直接在原 teacher 的 CUDA 12.8 环境里 `pip install -U`。Bootstrap 创建单独环境；CPU preflight 用 `readelf` 检查 TE、Apex、FlashAttention 的 ELF **version needs**，要求不高于运行主机和 ABCI compute glibc 2.34 中的较低版本，并保存二进制 SHA256。它不加载 GPU，不验证 GLIBCXX、CUDA 或全部动态依赖；GPU preflight 继续检查真实扩展导入、BF16 运算和显存。ABCI driver 与该 CUDA 13 栈是否兼容仍需实机验证。

两个正式 arm 在 200 步内共创建 80 份完整 checkpoint，但滚动保留每组最近 2 份和 validation 最佳，正常最多各 3 份；最佳在最近两份内时不重复保存。step200 终点属于最近两份，始终保留。另保留 4 份 qualification checkpoint 作为恢复证据。CPU preflight 分别估算稳态保留空间和两组各新增一份尚未清理时的峰值空间，按 BF16 权重×7 估算完整 Adam 训练状态；不包含中断留下的半成品，也不能验证项目 quota，需结合 ABCI quota 核对。

Teacher 推理 `max_model_len` 为 padding 上界与 prompt-logprob API 预留一个 token；实际每条 student prompt+response 仍受冻结的 65,536 token 总预算限制。

## 6. 三层验收

| 层级 | 检查内容 | 当前证据 |
|---|---|---|
| 本地 CPU 软件测试 | 官方 Hydra/dataclass、原生 loss 与梯度、原生 teacher routing、数据 factory、token 对齐、采样/loader 恢复、checkpoint 与 PBS 状态机 | `tests/mopd` 已通过，见下面复跑命令 |
| ABCI CPU preflight | 实际导出/数据/分割/tokenizer/verifier/配置，以及原生 ELF glibc 要求 | 原 v3 PASS 未包含 ABI 检查；修复后必须重新执行 |
| ABCI GPU qualification | 真实四 teacher scoring、Megatron forward/backward/update、原生完整 checkpoint；step4 正常退出，再由新 PBS 进程恢复到 step8 | 两组均在 TE 导入失败；尚无通过记录 |

Qualification 使用 **B=8、8 updates、response 上限256**，从每域最长的两个训练 prompts 取测试样本；调试评估每域一个样本。这些数值是本项目适配层新增的 smoke/reload 调试设置，来自工程实现，不来自原始 MOPD 论文，也不属于正式训练超参数。路径位于 `qualification/`，不参与正式选模。它验证基本执行与恢复，**不证明 B2048/64K 的最坏长度吞吐和显存容量**。正式运行出现 OOM 会停在 BLOCKED，不会改 batch 或截断长度来伪装通过。

CPU 中的 AdamW 小模型恢复测试检查保存状态后下一次更新完全一致；它不替代上述真实 MoE 重启测试。分布式 loss 的 CPU 测试模拟 DP 梯度平均，实际 NCCL/DP 通信留给 GPU qualification。

自动流程同时登记 mix 和 only 两条测试链，每条各自执行 step1–4、在新 PBS job 中恢复至 step8。两条链独立推进；一条链排队、恢复或 BLOCKED 不阻止另一条继续测试。两份结果都必须绑定本实验、对应 arm、step8 和 resumed_from=4，才能提交两组正式训练。正式任务从 base 初始化，使用独立的 `runs/` 目录，不继承 qualification 权重。无中断时共 4 个资格测试 job、2 个训练 job 和 2 个最终 test job。资格测试和正式训练阶段均最多并发 4 节点/32 GPU；同一 arm 的 step4 与恢复测试仍有先后依赖，不会四个资格 job 同时运行。实际启动时间由 PBS 可用资源决定。明确的应用错误 BLOCKED、STOP 或尚未确认的 PBS 状态均不会被当成通过。

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
mopd_output="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3"

"$mopd_python" -I scripts/mopd.py prepare \
  --teacher-campaign "$mopd_teacher" --output "$mopd_output" \
  --verl-src "$mopd_verl" --python "$mopd_python"
"$mopd_python" -I scripts/mopd.py preflight --manifest "$mopd_output/manifest.json"
)
```

注册自动流程并启动监护。这个入口已包含测试通过后的正式训练及最终 test，不需要等待测试结束后再手动提交：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3/manifest.json"
"$mopd_python" -I scripts/mopd.py start --manifest "$mopd_manifest" --phase pipeline --arm both
"$mopd_python" -I scripts/mopd.py status --manifest "$mopd_manifest"
timeout 20s qstat -u "$USER" || true
)
```

流程意图写入 `pipeline.json`，重复执行同一命令或执行 `resume-control` 会接管原任务；不会重复提交已登记的 job。仍可使用 `--phase qualification` 登记资格测试：没有 pipeline 时不会自动启动正式训练；已有 pipeline 时保留其后续自动调度。也可在对应测试已通过后用 `--phase train` 单独注册正式训练。`--arm domain_mix` / `domain_only` 可限制流程只包含一个 arm；已注册 pipeline 的 arm 集合不能通过重复 start 改写。后面的简短状态/暂停命令需要在登录 shell 中重新设置 `mopd_python`、`mopd_manifest` 并进入 checkout；上面的 subshell 不会保留这些变量。

`...val1024_roll2_v3` 已冻结自动流程及滚动保留代码。更新 checkout 不会改变旧 manifest 内的 runtime。对于已经用 `63d59e2` 启动、只有 Mix 入队的 v3 实验，冻结监护本来就会处理共享状态表内的所有任务，因此可以直接补登记 Only，使两条测试链并行。下面的命令通过启动锁及状态锁保留原 Mix 的 job/pending、已有 CPU preflight 和 pipeline；复用现有监护，两组通过后仍自动进入正式训练。不需要重建环境、重新 prepare 或替换冻结 runtime：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
git fetch --no-tags origin codex/d4-mopd-baseline
git merge --ff-only FETCH_HEAD
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3/manifest.json"
"$mopd_python" -I scripts/mopd.py start --manifest "$mopd_manifest" --phase pipeline --arm both
"$mopd_python" -I scripts/mopd.py status --manifest "$mopd_manifest"
timeout 20s qstat -u "$USER" || true
)
```

重复执行不会再创建同一任务。登记完成时 Only 可能还是 READY；监护下一轮才提交到 PBS。`automatic_pipeline` 返回已有自动流程是否启用，不取决于本次 start 的 phase。

## 8. Checkpoint、监护和中断

- 原生 checkpoint 必须包含 model、optimizer、LR scheduler、worker RNG，再加 DataLoader state 和 driver RNG。所有文件完成后才原子写入 `mopd_complete.json`。
- 滚动保留先从全部 validation 收据重建最佳模型指针，再校验完整状态。最近 2 份、validation 最佳、固定预算终点、已冻结最终 test 使用的 checkpoint 都受保护。step0 最佳时直接保留 base 引用，不复制 base。不会清理原 teacher campaign。
- 旧 checkpoint 先原子移入该 run 的 `checkpoints/.retention_trash`，再删除；清理计划及原始 marker 写入 `checkpoint_retention/*.json`。若删除中断，恢复只扫描仍完整的 checkpoint，接管后完成待清理文件。`retention.json` 记录保留/已删除步数；训练日志、完整 validation 历史、生成结果不删除。新 checkpoint 尚未完整保存时不触发滚动清理。
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

## 9. 修复 2026-11-10 的 TE/glibc 启动失败

失败发生在 `gpu-node` 导入 `transformer_engine.pytorch` 时；两台计算节点均报告 `libtransformer_engine.so` 需要 `GLIBC_2.38`。PBS exit 1 作为应用错误保持 BLOCKED。`start` / `resume-control` 不清除该状态。用户后来中断 `sleep` 不影响此前已失败的独立 PBS job。

本次使用 NVIDIA TE v2.20.2 源码 commit `6ea2a74a9e98c99e6d7b164a33775cc457520027` 重建整个 monolithic wheel；遵循 [NVIDIA 源码安装方式](https://docs.nvidia.com/deeplearning/transformer-engine/installation.html)，并保持 [verl cu130 wheelhouse](https://github.com/verl-project/verl-wheelhouse/blob/main/ci/build_scripts/transformer_engine.sh) 的 PyTorch-only / `NVTE_WITH_NCCL_EP=0` 设置。使用 gcc13.2/CUDA13.0.1、现有环境的 cuDNN/NCCL 头文件，只编译 H200 的 SM90；编译并发为 1。已有依赖版本必须全部保持不变，只允许补入缺少的 wheel、pybind11 构建工具。先生成 wheel，再 `--no-deps --reinstall`；不会修改系统 glibc。

ABCI 的 CUDA 模块按层级命名，例如官方文档中的 `cuda/12.6/12.6.1`（[模块使用说明](https://docs.abci.ai/v3/en/environment-modules/)、[CUDA 主次版本表](https://docs.abci.ai/v3/en/tips/gds/)）。旧修复脚本误写了扁平的 `cuda/13.0.1`，会在编译前报 modulefile 不存在。现先探测 `cuda/13.0/13.0.1`，兼容确实存在的同版本扁平名称；不选择默认版本。加载后核验实际 `nvcc` 为 13.0，并固定 `CUDACXX`、`CUDA_HOME`。模块清单、探测输出、已加载模块及工具链记录分别保存在 `cuda_modules.txt`、`module_probe.log`、`loaded_modules.txt`、`toolchain.json`，后者也进入 wheel 收据。模块缺失或编译器版本不符时，在下载源码及安装前停止。更新 checkout 后重新运行下面的 `repair-te` 即可；此次模块加载失败没有安装 TE，也没有重新提交资格测试。

`repair-te` 可从任一 ABCI 登录节点运行，不需要原 guardian 所在主机。它持有共享控制锁、验证旧 PBS 作业已终止且资格测试没有 checkpoint，备份并作废旧 CPU PASS，记录源码、编译参数、二进制哈希和依赖清单；构建失败时不放行任务。也支持尚未提交任何任务的新 manifest。命令有现场编译过程，完成前请保持会话；如被中断，后续从同一命令重新构建，原任务记录保持 BLOCKED。源码构建 wheel 保存在实验的 `environment_repairs/<id>/wheels/`，详细输出在 `build.log`。构建成功不代表 GPU 验收成功。

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
git fetch --no-tags origin codex/d4-mopd-baseline
git merge --ff-only FETCH_HEAD
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3/manifest.json"
"$mopd_python" -I scripts/mopd.py repair-te --manifest "$mopd_manifest"
"$mopd_python" -I scripts/mopd.py preflight --manifest "$mopd_manifest"
"$mopd_python" -I scripts/mopd.py retry-qualification --manifest "$mopd_manifest" --arm both
"$mopd_python" -I scripts/mopd.py status --manifest "$mopd_manifest"
timeout 20s qstat -u "$USER" || true
)
```

`retry-qualification` 重验 ABI 文件哈希和版本、重新查询旧 job 的 PBS 终态，归档原 state 后一次性放行两个 arm；UNKNOWN、仍运行、已有 checkpoint、guardian 活跃或待确认 qsub 时拒绝重置，不 qdel，不清理旧 attempt。重新启动现有并行 pipeline：两组各完成 step4→新进程恢复到 step8 后，才自动开始正式训练。失败仍为 BLOCKED，不循环重试应用错误；271 等外部中断恢复策略不变。

本次保留 v3 的 manifest、冻结 runtime、模型、数据及训练设置；同版本 TE 的编译来源另存审计。新增的 GPU 失败 JSON 和 shell 早期退出记录只会进入**以后新 prepare 的 runtime**，不原地修改 v3 快照。当前 v3 通过新版 checkout 执行 CPU preflight 和显式重试，再由原有冻结 guardian 接管。修复后不要手工 `uv sync`，它会重新安装锁文件里的 wheelhouse 包；bootstrap 看到源码构建标记也会拒绝覆盖。新实验第一次提交可在修复、preflight 后使用普通 `start`。

## 10. 查看结果与测量时间

`status` 返回 guardian 锁、heartbeat、自动流程计划、每个任务及上次 PBS 观察，以及每个 arm 的 checkpoint 保留/删除记录。每个 `runs/<arm>/attempts/<id>/` 保留：

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

Math validation 同时报告 Nemotron、DeepMath、L8/L9 与两个来源等权均值。当前 validation 两来源样本数相同，按样本数加权与来源等权的 Math 均值相等；最终 test 仍保留 Nemotron 226、DeepMath 256，因此 test 上两种聚合口径仍需区分。Task verifier 用于 validation 和最终 test，通过原环境的隔离 Python 子进程运行；训练 loss 不调用它。

比较收敛时间时分开统计 optimizer update、generation、完整 step、validation/checkpoint 开销、队列等待及中断重做。不要把 `actor_update_seconds` 当成完整训练步耗时，也不要把重跑的同一步累计为新增训练步。中断 attempt 的 metrics 保留，不覆盖。

训练结束后自动对 validation 选出的 checkpoint 调度独立最终 test，流程见第 4 节。固定 test 共 1,250 题，不随本次 validation 抽样改变；结果写入 `final_tests/<arm>/receipt.json`。选模记录保存在 `runs/<arm>/validation/step_<n>.json` 和 `best_validation.json`，test 结果不用于选模、采样或 loss。

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

这些测试直接导入固定版本的上游 loss/config/teacher manager；缺少 upstream checkout 或导入失败会失败，不会用 skip 冒充通过。并行调度回归覆盖两种完成先后顺序、step4 新进程恢复、两组通过前禁止正式训练、已有冻结监护补登记 Only、pending 提交恢复，以及单组 271/265/查询超时/明确 FAIL 对另一组的隔离。

## 11. 文件导览

`config.py` 与 `compat.py` 接上游配方；`assets.py` 冻结真实资产；`spec.py` 定义两 arm 的数据计划；`integration.py` 在原生 trainer 边界检查并提交 checkpoint；`checkpoints.py` 负责完整性；`preflight.py` 分开 CPU/GPU 证据；`control.py`、`node.sh`、`cli.py` 负责 PBS；`metrics.py` 与 `reward.py` 保留来源级 validation。所有修改均在新增 `d4_mopd` 等文件内，未改原 teacher 控制器或上游 verl 源码。
