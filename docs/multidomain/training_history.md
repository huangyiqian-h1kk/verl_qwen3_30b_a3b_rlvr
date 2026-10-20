# Random/domain 训练历史汇总

`scripts/multidomain/collect_training_history.py` 使用 Python 标准库，读取所有历史训练作业，而非仅当前 job 或日志末尾。可以通过 `git show COMMIT:path | python -I -` 执行，无需切换或覆盖集群上的工作树。

默认覆盖以下四个 run，分别保留其身份：

| 流 | run 后缀 |
|---|---|
| random 单节点正式训练 | `qwen30b_d4_10k_uniform_ctx64k_seed42_v1` |
| domain 单节点正式训练 | 上述名称加 `_db_v1` |
| random 双节点候选／后续训练 | 上述名称加 `_n2_once_v1` |
| domain 双节点候选／后续训练 | 上述名称加 `_db_n2_once_v1` |

```bash
python3 -I scripts/multidomain/collect_training_history.py --root "$PWD"
```

结果保存在 `outputs/analysis/d4_training_history_<UTC时间>/`，同时生成同名 `.tar.gz`。终端打印摘要表与 `RESULT_BUNDLE=` 路径。分析文件和原始证据均在压缩包内，可直接上传该结果包。

文件内容：

- `summary.json`：各流已记录 update 的最大 step、step 覆盖、缺失／重复 step、validation step、所有指标的初末值和观测范围，以及训练步骤耗时、PBS 占用时长和 GPU 小时。
- `metrics.jsonl`、`metrics_long.csv`：完整 console 指标，包括 loss、reward、entropy、KL、梯度、学习率、输出 token 长度／截断率、各域 validation、rollout／reward／update／save／validation 耗时；仅导出实际记录过的键。每条附 run、job、phase、文件、行号和重启 segment。
- `generation_rows.jsonl`、`generation_summary.json`：逐输出的全部现有数值字段，以及按文件、step、domain 的均值／范围。无法从输出文件识别 domain 时记录 `unknown`。
- `output_samples.jsonl`：各文件、step、domain、得分类别最多两个输出示例。长文本保留前 12000 字符并标出原始字符数。这是便于查看的样本，不用于估算总体性能。
- `training_events.jsonl`：恢复、checkpoint、validation、domain 采样和 verifier 相关日志行。
- `jobs.json`：作业 request 和仍能查询到的 PBS 资源／时长记录。
- `checkpoints.json`：checkpoint 元数据清单；不读取模型或优化器张量，不据此宣称 checkpoint 完整可恢复。
- `manifest.json`、`warnings.json`：来源、日志快照 SHA256、扫描范围、变化／缺失／截断记录。
- 压缩包的 `evidence/`：完整作业文本日志、配置、恢复／交接记录、相关 reward／sampler／trainer 源码快照和 checkpoint 元数据。

Console 数值支持普通浮点数和 `np.float64(...)`、`numpy.int64(...)` 等数值标量表示；只按数值语法解析，不执行日志中的表达式，也不依赖 NumPy。

默认扫描 `generations/` 与 `raw_generations/` 的全部现有 JSONL 行，保留所有数值和代表性文本，不复制全部长回答。若需要所有回答全文，加 `--full-generations`；这可能明显增大压缩包。脚本不能恢复此前未保存的 rollout、token 数或采样统计，也不会重新调用 verifier/tokenizer 评分。

解释数据时必须保留的区分：

1. step 0 通常是初始 validation，不能补造 step 0 的训练 loss。复用的基线须依据 recovery/source 记录归属，不自动视为另一条实验重新评估过。
2. `Training Progress: 65/468` 或“开始 sampling step 66”不算完成第 66 步。最大 update step 依据实际 actor/update 指标行，durable checkpoint 另看证据。
3. 恢复后可能重跑相同 step。所有 attempt 均保留；不把重复 step 相加为净进度、不直接平均，也不将双节点候选拼接进正式曲线。`repeated_step_candidates` 包含同一指标的非完全相同日志镜像，需结合来源与恢复记录审查。
4. 指标初末值和范围覆盖所有观测 attempt；如果一个端点有多个值，完整列出，不擅自选择最终路径。reward/verifier 版本变更可由归档配置、源码哈希和恢复日志识别。
5. 步骤耗时使用 `timing_s/step`，优先每个 job 的 `pbs.log`，避免 phase/tee 镜像重复计时；没有主日志且存在多个指标来源时不合计该 job 的步骤耗时，并列出歧义。它包括已记录的重跑步骤，不等于净训练进度成本。
6. PBS 总时间使用 `resources_used.walltime`，包括初始化、验证、失败和重跑。GPU 小时乘以实际请求的 GPU 数；绝不将 `cput` 或 qstat 的 Time Use 当墙钟时间。历史 PBS 记录过期时明确显示覆盖数，合计不代表完整实验成本。
7. 共享输出路径在重跑时可能覆盖同一步的生成文件；本工具只能汇总收集时仍存在的文件。运行中的文件采用收集时快照／前缀，报告非原子性和不完整末行。
8. 单节点与双节点、smoke/check 与正式训练、train reward 与固定 validation、字符数与 token 数分别保留。未记录的 pass@k、分组正确率、entropy 等均不补造。
9. W&B 二进制历史仅建立路径清单；不导入 W&B、不联网同步。若 console 指标缺失，`warnings.json` 和 step 覆盖会显示采集缺口，可另从指定 W&B run 导出补齐。

CPU 回归：

```bash
python3 -m unittest discover -s tests/multidomain -p 'test_training_history.py' -v
```

脚本只查询 `qstat`，不提交、停止、修改或重启作业；不加载训练库或 GPU。`--no-pbs` 可跳过调度器查询。输出目录必须不存在，避免覆盖上次汇总。
