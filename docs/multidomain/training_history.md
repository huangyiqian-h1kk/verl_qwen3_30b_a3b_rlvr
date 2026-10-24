# Random/domain 训练历史汇总

`scripts/multidomain/collect_training_history.py` 使用 Python 标准库，读取所有历史训练作业，而非仅当前 job 或日志末尾。可以通过 `git show COMMIT:path | python -I -` 执行，无需切换或覆盖集群上的工作树。

`2026-10-24.history.3` 默认覆盖以下六个 run，分别保留其身份，并合并独立补评任务：

| 流 | run 后缀 |
|---|---|
| random 单节点正式训练 | `qwen30b_d4_10k_uniform_ctx64k_seed42_v1` |
| domain 单节点正式训练 | 上述名称加 `_db_v1` |
| random 双节点候选／后续训练 | 上述名称加 `_n2_once_v1` |
| domain 双节点候选／后续训练 | 上述名称加 `_db_n2_once_v1` |
| random 接续训练，包括成功交接后的双节点 | 上述名称加 `_continuation_v1` |
| domain 接续训练，包括交接失败后继续的单节点 | 上述名称加 `_db_continuation_v1` |

接续训练同时扫描 `jobs/` 与 `lineages/<lineage>/`。候选作业的 A、B、C、D、E 阶段和正式作业的 `train` 阶段均保留；E 阶段可以在交接后继续正式训练，不能将整个 candidate 文件夹当成一次性 check 丢弃。

```bash
python3 -I scripts/multidomain/collect_training_history.py --root "$PWD"
```

结果保存在 `outputs/analysis/d4_training_history_<UTC时间>/`，同时生成同名 `.tar.gz`。终端打印摘要表与 `RESULT_BUNDLE=` 路径。分析文件和原始证据均在压缩包内，可直接上传该结果包。

文件内容：

- `training_dynamics.csv` / `.jsonl`：两套 setup 的训练指标统一表，覆盖 loss、reward、entropy、KL、学习率、输出长度、各域批次统计和计时等实际记录。`source_kind=console/structured` 区分日志与结构化证据，同一更新可能有两种观测，不能把两行相加为两步。`formal_attempt`、`step_usage`、job、phase、lineage 保留重跑和候选分支信息。
- `validation_dynamics.csv` / `.jsonl`：统一 validation 表，包含训练内评估和历史补评。横坐标是被评 checkpoint 的 `step`，`at_utc` 是完成评估的时间。Random 125、domain 100 的补测归回 125/100，不放到补评提交时的最新训练步。保留四域 `val-core/*/acc/mean@1` 和全部现有 val-aux 指标。
- `validation_backfills.json`：扫描 `outputs/multidomain/validation_backfills/*` 的 campaign 和全部 PBS 提交记录，并绑定两套 setup 各自的 evaluation 子目录。重复续投和复用已完成结果均保留来源。
- `validation_progress.json`：新评估协议的已完成批次、行数与本进程复用批次数；只归档 manifest/progress，默认不重复打包大量 batch 缓存文本。
- `job_timeline.csv`：训练与独立补评作业的 job ID、恢复起点、节点数、退出码、PBS comment、墙钟时长及 GPU 小时。一个补评 PBS allocation 下的两个子进程只计一次共享资源成本，放在 `summary.json.shared_backfill_accounting`，不分别加到两套训练成本中。
- `summary.json`：`setups` 汇总 random/domain 正式训练 attempt 的已完成 step 和当前监护记录；`roles` 保留六个来源各自的 console update、结构化已完成 step、缺失／重复 step、validation、指标初末值、步骤耗时、PBS 占用时长和 GPU 小时。最大观测 step 不等于已经解析好回退关系的最终有效进度。
- `metrics.jsonl`、`metrics_long.csv`：完整 console 指标，包括 loss、reward、entropy、KL、梯度、学习率、输出 token 长度／截断率、各域 validation、rollout／reward／update／save／validation 耗时；仅导出实际记录过的键。每条附 run、job、phase、文件、行号和重启 segment。
- `step_records.jsonl`、`step_metrics_long.csv`：`step_<n>.json` 的完整结构化记录及数值，包括实际 prompt/response 数、各域 response 数、reward、response token 长度和 actor 指标。附 lineage、分配节点数、阶段训练 rank 数、execution ID、完成标记及交接分类。JSON 保留完整 prompt ID 列表与 actor 原始样本；只在 `status=PASS`、actor update 完成、weights 同步三个条件都满足时标记已完成 update。
- `continuation_status.json`：两套接续控制器的 state 和持久化交接记录快照。读取时刻可能不同于 PBS 查询时刻，不代替实时状态查询。
- `generation_rows.jsonl`、`generation_summary.json`：逐输出的全部现有数值字段，以及按文件、step、domain 的均值／范围。无法从输出文件识别 domain 时记录 `unknown`。
- `output_samples.jsonl`：各文件、step、domain、得分类别最多两个输出示例。长文本保留前 12000 字符并标出原始字符数。这是便于查看的样本，不用于估算总体性能。
- `training_events.jsonl`：恢复、checkpoint、validation、domain 采样和 verifier 相关日志行。
- `jobs.json`：作业 request 和仍能查询到的 PBS 资源／时长记录。
- `checkpoints.json`：checkpoint 元数据清单；不读取模型或优化器张量，不据此宣称 checkpoint 完整可恢复。
- `manifest.json`、`warnings.json`：来源、日志快照 SHA256、扫描范围、变化／缺失／截断记录。
- 压缩包的 `evidence/`：完整作业文本日志、配置、恢复／交接记录、相关 reward／sampler／trainer 源码快照和 checkpoint 元数据。

Console 数值支持普通浮点数和 `np.float64(...)`、`numpy.int64(...)` 等数值标量表示；只按数值语法解析，不执行日志中的表达式，也不依赖 NumPy。

补评验收与去重：

- 新版 `control/evaluations/*.json` 的记录只有在 producer/consumer 请求身份、setup、lineage、评估 protocol、checkpoint 元数据或 phase proof、四域有限数值、完整 1006 行输出及 SHA256 均匹配时，才标记 `verified_complete`。有 journal 时也检查 manifest 绑定。这是评估证据校验，不读取或重新校验模型张量。
- 已被滚动清理的训练 checkpoint 可以用仍保留的 `validation_<step>.json` 对 receipt 的 SHA256 绑定验收；不能因此宣称那个 checkpoint 的张量仍可恢复。
- 同一 `(setup, checkpoint_key, protocol)` 的复用结果只保留一个评估点，保留 `consumer_jobs` 和重复来源。不同 verifier/protocol 或不同 checkpoint 的相同步数不自动合并、不平均。相互冲突的 receipt 标记 `conflicting_receipts`。
- 老日志里的评估保留为 `console_reported`，不冒充完整输出验证。缺文件、SHA 不符、部分输出、protocol 不符等保留 `unverified_receipt` 和具体错误。原日志仍在原始 metrics 表和 evidence 内。
- evaluation-only 子进程显式标为 `stage=evaluation`，其日志或 checkpoint step 不增加训练步数。`backfilled_validation_steps` 只列验收通过的补测；`scheduled_steps_without_observed_metrics` 表示采集证据缺口，不自动判定训练程序没有执行评估。

这两个 dynamics CSV 是保留来源的观测表，可直接按 setup、step、metric 作后续分析；不是自动裁定过全部回退关系的唯一训练曲线。训练分析应选择明确的来源和正式分支，不能把 console 与 structured 指标、重复恢复 attempt 混为独立更新。

结构化 actor 数值标量放在 `actor_recorded/` 下。若某个 actor 指标是数值列表，额外导出明确命名的 `actor_samples/<原键>/mean` 和 `/count`，均值为这些已记录样本的算术平均，不冒充按 token 加权的 console 指标。原列表完整保留。批次统计放在 `batch/` 下；例如 `batch/reward_mean`、`batch/response_length_mean`、`batch/response_domain_counts/math`。这两套指标表是同一步的不同证据来源，不能相加为两次训练。

默认扫描 `generations/` 与 `raw_generations/` 的全部现有 JSONL 行，保留所有数值和代表性文本，不复制全部长回答。若需要所有回答全文，加 `--full-generations`；这可能明显增大压缩包。脚本不能恢复此前未保存的 rollout、token 数或采样统计，也不会重新调用 verifier/tokenizer 评分。

解释数据时必须保留的区分：

1. step 0 通常是初始 validation，不能补造 step 0 的训练 loss。复用的基线须依据 recovery/source 记录归属，不自动视为另一条实验重新评估过。
2. `Training Progress: 65/468` 或“开始 sampling step 66”不算完成第 66 步。console update 和结构化已完成 update 分别记录，setup 概览对二者的 step 取集合并集；durable checkpoint 另看证据。
3. 恢复后可能重跑相同 step。所有 attempt 均保留；不把重复 step 相加为净进度、不直接平均，也不将双节点候选拼接进正式曲线。`repeated_step_candidates` 包含同一指标的非完全相同日志镜像，需结合来源与恢复记录审查。
4. 指标初末值和范围覆盖所有观测 attempt；如果一个端点有多个值，完整列出，不擅自选择最终路径。reward/verifier 版本变更可由归档配置、源码哈希和恢复日志识别。
5. 步骤耗时使用 `timing_s/step`，优先每个 job 的 `pbs.log`，避免 phase/tee 镜像重复计时；没有主日志且存在多个指标来源时不合计该 job 的步骤耗时，并列出歧义。它包括已记录的重跑步骤，不等于净训练进度成本。
6. PBS 总时间使用 `resources_used.walltime`，包括初始化、验证、失败和重跑。GPU 小时乘以实际请求的 GPU 数；绝不将 `cput` 或 qstat 的 Time Use 当墙钟时间。历史 PBS 记录过期时明确显示覆盖数，合计不代表完整实验成本。
7. 共享输出路径在重跑时可能覆盖同一步的生成文件；本工具只能汇总收集时仍存在的文件。运行中的文件采用收集时快照／前缀，报告非原子性和不完整末行。
8. 单节点与双节点、smoke/check 与正式训练、train reward 与固定 validation、字符数与 token 数分别保留。未记录的 pass@k、分组正确率、entropy 等均不补造。
9. W&B 二进制历史仅建立路径清单；不导入 W&B、不联网同步。若 console 指标缺失，`warnings.json` 和 step 覆盖会显示采集缺口，可另从指定 W&B run 导出补齐。
10. random 和 domain 的交接各自认定。接续候选只有在 `committed.json` 的 identity、job ID、lineage 与 request 相符且含 checkpoint step 时，才进入该 setup 的正式 attempt 汇总；该判断是已记录交接元数据的绑定，不重新执行 GPU 恢复验证。未交接候选仍保存在原始指标和六个 role 的全 attempt 汇总中。
11. `step_usage` 区分 qualification、catch-up/uncommitted、formal 和 formal-after-handoff。成功候选的 A–E 验证步骤属于已接受分支的历史，不能再次算成额外净进度。交接点前后以及回退后重跑的相同步数仍须按来源审核，脚本不擅自选择唯一曲线。
12. `allocation_nodes` 是 PBS 请求节点数；`target_world` 是该阶段实际训练的 rank 数。双节点候选在 C/D 的八 rank 验证阶段仍占用两个节点，计算资源成本时不能按单节点收费量估算。

从已 fetch 的固定 commit 执行时，可以保持正在运行的监护工作树不动：

```bash
# D4_REV 使用交接命令给出的完整 commit SHA；D4_OPS 为已有运维工作树。
git -C "$D4_OPS" show "$D4_REV":scripts/multidomain/collect_training_history.py |
  "$D4_PY" -I - --root "$D4_ROOT"
```

CPU 回归：

```bash
python3 -m unittest discover -s tests/multidomain -p 'test_training_history.py' -v
```

脚本只查询 `qstat`，不提交、停止、修改或重启作业；不加载训练库或 GPU。`--no-pbs` 可跳过调度器查询。输出目录必须不存在，避免覆盖上次汇总。
