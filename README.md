# Data Mixture RL

本分支 `research/data-mixture-rl` 用于 Qwen3 多领域 GRPO 实验。ABCI 项目目录为：

```text
/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl
```

`initial_setup_v1/` 是上一轮 ZIP 解压得到的独立准备工具包。你已在其中通过 9 项准备单测和 20 项上游评分契约测试。本分支的新入口从**项目根目录**运行，见下面的命令。

## 代码与运行目录

| 路径 | 用途 |
|---|---|
| `config/multidomain/initial.yaml` | 域开关、模型路径、并行配置、训练参数、PBS 队列 |
| `config/multidomain/sources.lock.json` | 数据、tokenizer、模型和恢复依赖的固定 revision |
| `multidomain/adapters.py`、`template.py` | 原始数据转换、Qwen 消息/工具模板、下一次调用解析 |
| `multidomain/build.py` | 去重、关联样本分组、真实 token 计数、固定划分、配平、Parquet 导出 |
| `multidomain/reward.py`、`_vendor/` | 统一 reward；固定版本的官方评分源码及哈希 |
| `multidomain/dataset.py`、`agent_loop.py` | verl Dataset 和单轮生成；逐条工具 schema 与输出长度上限 |
| `multidomain/runtime_config.py`、`train.py`、`worker.py` | 固定 verl 接入、正式训练、无更新 backward、checkpoint 保存 |
| `scripts/multidomain/submit.sh` | 统一 PBS 提交入口；支持 `--dry-run` |
| `tests/multidomain/` | CPU 回归测试 |
| `data/raw/` | 固定版本原始数据及下载清单；运行时生成 |
| `data/multidomain/all_six_v1/` | 本轮数据、索引、保留池、清单；运行时生成 |
| `outputs/multidomain/<run-id>/` | 作业日志、验收报告、生成结果、checkpoint；运行时生成 |
| `docs/legacy/README_before_multidomain.md` | 旧项目 README |

旧实验脚本保留。共享 verl checkout 为原项目的 `src/verl`，必须位于 `62ac6d6e9fccc09ab6af2171217cee6e290b799b`；新代码通过 `PYTHONPATH` 接入。旧模型目录、旧实验输出和本轮输出各有独立路径。

## 实验配置

- 必选域：Math、IF。默认另启用 Science、Conversational Pivot、SWE Pivot、Logic/Algorithmic。
- IF 包含 free-form、citation 和 structured 三个来源；structured 同时支持文本格式与工具参数提取。
- Science 只取原生无工具 agent。冻结 Qwen3-235B-A22B-Instruct-2507-FP8 judge，单副本 TP8、独占 8 GPU。
- 策略模型 Qwen3-30B-A3B-Instruct-2507，单节点 8 GPU，actor TP2/PP2/EP2；rollout TP4、两个副本。
- 256 prompts × K8，GRPO；学习率 `1e-6`；3 epochs；PPO minibatch 32；entropy 和 KL 均关闭。
- 每条 prompt 与 response 合计最多 65,536 tokens。原始 `max_output_tokens` 更小时保留该上限。实际 Qwen 模板计数；超长数据隔离；不截断历史。
- 全域统一建立泄漏关联分组，再用 seed 42 划分 90/5/5。每个启用域抽取相同的 eligible train 数量。IF 来源按容量封顶等额抽样；RG 保持类别比例。
- Validation/test 每域最多 256 条；域开关不会改变其他域固定评估样本。未选中的 held-out 样本仍保留在 held-out 池。
- 工具任务仅预测下一个 assistant turn，比较调用或消息；不执行生成的工具或代码。公开的历史 reasoning summary 转为 assistant 文本保留，原始请求控制项写入 provenance。
- Pivot 评分保持上游行为，包括 SWE 多词参数宽松比较、message 类型匹配即可得分，以及只取第一条调用。没有额外格式奖励。
- 保存频率和验证频率均为 25 steps；保留最近 2 个完整续训 checkpoint；最后一步额外导出 HF 权重。没有每步权重快照或 watcher。

RG 的两个已发现异常的 propositional_logic 参考答案按 UUID 隔离，原因进入数据报告。这是已知数据问题的隔离。未添加基于模型成功率的筛选。

## 当前验收边界

本分支实现了下面各阶段的入口。CPU 验证记录在 `docs/multidomain_validation.json`。这些结果覆盖解析、数据结构、评分适配和配置；ABCI 的 NCCL、模型加载、judge 质量、64K 显存、正式训练及断点续训仍需实际作业验收。作业报告只有在相应命令成功结束后才写为 PASS。

`backward` 使用真实 actor engine 完成 token NLL forward/backward，不调用 `optimizer.step`。它验证该长度下的前后向容量。正式训练的首次 optimizer step 另写每个 rank 的显存峰值；不能用 backward 的峰值替代。

## 1. 应用补丁并准备 Python 依赖

初版通过 `DATA_MIXTURE_RL_V1_2026-09-26.patch` 交付，目前已提交到 `research/data-mixture-rl`。已应用补丁的工作目录不要重复应用。以下命令从现有项目根目录更新该分支并安装依赖。

在 ABCI 登录节点执行：

```bash
(
set -e
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl
git status --short
git switch research/data-mixture-rl
git pull --ff-only origin research/data-mixture-rl

source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate /groups/gcg51557/experiments/0390_rlsd/envs/verl_qwen3_moe_megatron_py312_cu128

bash scripts/multidomain/setup_cpu_dependencies.sh
source scripts/multidomain/env.sh
python -m unittest discover -s tests/multidomain -v
)
```

应当位于 `research/data-mixture-rl`。外层括号在独立子 shell 内执行：任一步失败即停止，不改变当前登录 shell 的退出选项。切换分支或拉取报错时保留输出和工作文件，不要运行强制 reset。

依赖脚本在 `.venv-multidomain/` 建立可以读取原 conda 包的独立环境，用原 conda 解释器的 `pip list --format=freeze` 生成 `包名==版本` 约束，包括 editable 安装的包。训练框架与共享依赖继续使用这些约束；脚本中明确列出的评分包及其专用依赖可以在项目虚拟环境中另装版本。例如原 conda 的 `math-verify 0.9.0` 保留，新实验使用 `0.8.0` 及其要求的 `latex2sympy2_extended 1.10.2`。安装失败时可直接重跑，无需删除环境。其余依赖冲突仍会停下；共享 conda 的包不会被修改。后续每个 PBS 作业会自动加载这个环境。

旧版脚本若报 `Editable requirements are not allowed as constraints`，说明依赖安装尚未完成；此时后续的 `xmltodict` 缺失是安装失败的结果。修订后的安装器记录原环境基线，只允许已确认的 outlines/Megatron/decord 诊断保留；新增问题仍失败。CPU 依赖成功状态为 `CPU_DEPENDENCIES_READY` 或 `CPU_DEPENDENCIES_READY_WITH_INHERITED_CONFLICTS`，随后应有 33 项测试全部 `OK`。这不代表 GPU 环境验收完成。版本来源、未解决问题及输出位置见 [依赖说明](docs/multidomain_dependencies.md)。提前执行 Git commit/push 不会引起安装错误，无需撤销提交。`env.sh` 保留调用者原有的 shell 选项，PBS 启动脚本仍自行开启严格模式。

## 2. 准备固定版本数据和模型资产

以下下载命令需要联网。在有网络的登录节点或下载机器运行；正式 compute 作业使用离线模式。

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl
source scripts/multidomain/env.sh

python -m multidomain.preparation.fetch_raw \
  --config config/multidomain/initial.yaml \
  --output-dir data/raw
```

此命令下载八个固定版本来源，恢复 Math 的 DAPO/Skywork 占位题，并生成 SHA256 清单。它没有启动训练，也没有用工具包中的候选数代替最终训练样本数。为使后续域消融共用全局泄漏分组，原始语料始终下载完整八个来源。

策略模型已有目录时仍需核对 tokenizer；Science 数据准备还需要 judge tokenizer。下面只下载配置/tokenizer，保留已有权重：

```bash
python -m multidomain.assets --model policy \
  --destination /groups/gcg51557/experiments/0390_rlsd/models/Qwen3-30B-A3B-Instruct-2507

python -m multidomain.assets --model judge \
  --destination /groups/gcg51557/experiments/0390_rlsd/models/Qwen3-235B-A22B-Instruct-2507-FP8
```

首次模型验收前必须具备完整权重。如果 judge 权重尚未下载，执行：

```bash
python -m multidomain.assets --model judge --weights \
  --destination /groups/gcg51557/experiments/0390_rlsd/models/Qwen3-235B-A22B-Instruct-2507-FP8
```

策略权重不完整时，同样在 policy 下载命令中加 `--weights`。已有完整权重时，用 `--verify-existing-weights` 校验固定 revision 的官方 SHA256，无需重新下载权重：

```bash
python -m multidomain.assets --model policy --verify-existing-weights \
  --destination /groups/gcg51557/experiments/0390_rlsd/models/Qwen3-30B-A3B-Instruct-2507
```

已存在的 judge 权重同理校验。模型阶段要求校验通过的资产清单。`--weights` 包括全部模型分片，下载量较大。下载资产清单记录 revision 和文件校验和；目录名称本身不能证明权重版本。

## 3. 在计算节点准备训练数据

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl
source scripts/multidomain/env.sh

bash scripts/multidomain/submit.sh \
  --stage prepare-data \
  --config config/multidomain/initial.yaml \
  --data-id all_six_v1 \
  --run-id qwen30b_d6_uniform_ctx64k_seed42_v1
```

入口打印 PBS 文件、job ID 和日志路径。添加 `--dry-run` 可查看完整 PBS 而不提交。已有非空数据目录不会覆盖；失败后先检查该作业日志和 `reports/prepare-data.json`，修复后使用新 data ID。

成功后，`data/multidomain/all_six_v1/` 包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 最终域计数、排除原因、tokenizer 指纹、数据哈希、实际配平规模 |
| `train.parquet` | 正式训练数据 |
| `validation.parquet`、`test.parquet` | 固定 held-out 数据 |
| `smoke.parquet` | 每域最多 8 条固定随机样本和 4 条最长样本 |
| `baseline_train.parquet` | 每域最多 64 条训练集基线样本 |
| `canonical.sqlite`、`index.jsonl` | 规范化记录、关联分组、长度预算、全部保留样本索引 |
| `excluded.jsonl` | 去重与长度隔离明细；原始行可由 provenance 找回 |

## 4. 分阶段验收与训练

后续命令都在同一项目根目录，使用同一个 `--config`、`--data-id` 和 `--run-id`。每个阶段结束并确认 PASS 后执行下一阶段：

```bash
bash scripts/multidomain/submit.sh --stage check-verifiers
bash scripts/multidomain/submit.sh --stage check-infra
```

默认 ID 就是上面的 `all_six_v1` 和 `qwen30b_d6_uniform_ctx64k_seed42_v1`。配置或代码变化会使验收报告失效；入口提示需要重跑的阶段。

| 阶段 | 实际操作 | 通过后得到什么 |
|---|---|---|
| `check-verifiers` | 每个非 judge 来源至少 32 条真实数据；额外覆盖 RG task 与 structured 格式；适配器结果对照固定上游函数 | `verifiers_detail.json`、CPU 测试结果 |
| `check-infra` | 2 节点 × 8 GPU、Ray 资源分配、3 个尺寸的跨节点 NCCL all-reduce | `infrastructure_detail.json`；不加载模型 |
| `judge-calibration` | 128 条人工标注候选；64 条等价、64 条不等价；仅加载冻结 judge | accuracy ≥95%、false-positive rate ≤5% |
| `smoke` | 固定 smoke 样本 K2，走完整模型→Qwen parser→reward；不更新参数 | 原始生成及评分日志；验证运行链路 |
| `backward` | 最长真实 prompt 加一个 token，以及 64K 容量样本；真实 actor engine forward/backward | 每 rank 显存报告；optimizer steps 为 0 |
| `baseline` | 训练子集 K8 随机采样；再对固定 validation 做 K1 确定性评估 | 训练前两套基线；不更新参数 |
| `train` | 正式 GRPO、周期性 validation、完整 checkpoint、最终 HF 导出 | 本轮训练结果与续训状态 |
| `evaluate` | 加载最终 HF 权重，对固定 test 做 K1 确定性评估 | 最终测试结果 |

Science 启用时，准备数据会生成：

```text
outputs/multidomain/qwen30b_d6_uniform_ctx64k_seed42_v1/review/science_128.jsonl
```

这是待人工填写的校准表。每题的两条记录用于填写等价与不等价候选。检查问题、参考答案和原生 `output_regex`，填写完整候选文本、`human_label`（0 或 1）、`reviewer`；总计必须恰好 64 个 1 和 64 个 0。标签默认空白。空标签、空候选或缺少 reviewer 都会被拒绝。

```bash
bash scripts/multidomain/submit.sh --stage judge-calibration \
  --calibration-file outputs/multidomain/qwen30b_d6_uniform_ctx64k_seed42_v1/review/science_128.jsonl

# 每次等待上一阶段成功后再提交下一条。
bash scripts/multidomain/submit.sh --stage smoke
bash scripts/multidomain/submit.sh --stage backward
bash scripts/multidomain/submit.sh --stage baseline
bash scripts/multidomain/submit.sh --stage train
bash scripts/multidomain/submit.sh --stage evaluate
```

Smoke 的 PASS 表示完整运行链路结束；生成质量和各域成功率需结合日志判断。Structured 的正例约束、judge 人工标签的质量、模型是否适合该任务不会由“进程成功退出”自动证明。

`raw_generations/` 保存 smoke、baseline、evaluate 中带工具标记的原始 completion 与 sample ID。`generations/` 保存 verl 的评分输出，域级指标进入训练日志。Science 请求超时、非法裁判标签、上下文溢出和数学评分进程失败会终止作业并记录错误，不会记成模型的 0 分。

## 续训和域消融

同一 run ID 再次提交 `train` 会从该目录最新完整 checkpoint 恢复模型、optimizer、scheduler 和数据加载状态。checkpoint 不兼容、丢失或验收失败时必须先处理错误。

改变域开关时，复制配置文件并修改 `domains.<name>.enabled`，保留必选 Math/IF；使用新的 data ID 和 run ID 重跑数据准备及验收。共享同一套 raw 文件与固定版本，所以其他域的 held-out IDs 保持稳定。每轮训练规模由该轮 eligible train 的最小域决定。

## Git 操作边界

本轮代码应应用并提交到原 repository 的 `research/data-mixture-rl` 分支；远端提交目前受 GitHub 连接器写权限阻塞。ABCI 的 `data_mixture_rl` 是该分支的独立工作目录。数据、模型、日志、checkpoint 和旧 `initial_setup_v1/` 不加入 Git。可通过 branch/commit、data ID 和 run ID 对照每轮实验。
