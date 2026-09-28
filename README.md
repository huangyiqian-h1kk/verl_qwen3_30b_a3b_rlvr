# 0390：四域单节点 GRPO

本分支使用 `math`、`if`、`conversational_pivot`、`logic_algorithmic`，各占训练集的 25%。关闭 science 和 swe_pivot，关闭 science judge。

## 环境与目录

- 项目目录：`/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d`
- 配置：`config/multidomain/four_domain.yaml`
- 数据：`data/multidomain/four_domains_v1`
- 运行：`outputs/multidomain/qwen30b_d4_uniform_ctx64k_seed42_v1`
- 原六域目录：同级 `data_mixture_rl`，原数据、报告和 checkpoint 保留。
- 复用原 Conda、项目 `.venv-multidomain` 和模型目录；不要在共享环境重新安装依赖。

## 实验参数

PBS 使用 `gcg51557` / `R9920261000` / `RTYPE=rt_HF` / `select=1`，作业名以 `0390_d4_` 开头。PBS 自动分配节点，不限定主机。

Policy 为 Qwen3-30B-A3B-Instruct-2507，1 节点 8 GPU；actor TP2 / PP2 / EP2，rollout TP4，共 2 个 rollout 副本。judge 不加载、不分配 GPU。

保留原六域的 GRPO 参数：seed 42，batch 256，rollout n=8，温度 1，学习率 1e-6，3 epochs，共享上下文上限 65,536 tokens，每 25 步保存，保留 2 个完整 checkpoint。

数据从原 `all_six_v2_ep8` 的完整 `canonical.sqlite` 与 `index.jsonl` 重选。沿用原全局分组、train/validation/test 划分、tokenizer 和长度判定；每域取最小训练候选池容量。IF 三个来源等量；logic_algorithmic 保留候选池类别比例。四域的 validation/test ID 与六域相同。预计训练集每域 6,621 条，共 26,484 条，实际以新 manifest 为准。

## 执行顺序

以下命令均在本项目目录执行。

```bash
bash scripts/multidomain/submit_four_domain.sh accept-four
```

一个单节点 PBS 作业内依次执行 `prepare-data`、`check-verifiers`、`check-infra`、`smoke`。先检查 `/dev/shm`，通过后启动 Ray。数据重选不重新下载或分词；verifier 检查使用真实四域数据；infra 验证单节点 8 GPU；smoke 每域最多 8 个随机样本加 4 个最长样本，去重后每条生成 2 个回答，不更新参数。

该命令不会启动正式训练。结尾必须出现 `FOUR-DOMAIN ACCEPTANCE PASS; optimizer_steps=0`，并检查 PBS `Exit_status = 0`。失败后保留日志；再次执行会复用本四域配置和代码对应的已有 PASS 阶段。数据准备若中断并留下不完整目录，不自动删除或覆盖，需检查后使用新的 data ID。

验收通过后，每一步完成并得到 PASS，再提交下一步：

```bash
bash scripts/multidomain/submit_four_domain.sh backward
bash scripts/multidomain/submit_four_domain.sh baseline
bash scripts/multidomain/submit_four_domain.sh train
bash scripts/multidomain/submit_four_domain.sh evaluate
```

- `backward`：前向/反向容量检查，不调用 optimizer.step。
- `baseline`：初始模型的训练子集和 validation 评估。
- `train`：正式 GRPO 更新；只使用四域训练集。
- `evaluate`：对最终 checkpoint 执行保留测试集评估。

不要一次性把以上四条全部提交。训练入口检查前序报告，防止跳过容量测试和 baseline。

## 代码入口

- `multidomain/reselect.py`：验证父数据、重选完整候选池、导出四域 parquet 和新 manifest。
- `multidomain/submit.py`：PBS 请求、单节点资源和日志路径。
- `multidomain/stage.py`：阶段执行、报告指纹及前序验收检查。
- `multidomain/runtime_config.py`：将实验参数映射到固定版本 verl；关闭 reward model 资源池。
- `scripts/multidomain/node.sh`：共享内存检查、每节点环境检查、Ray 启停。
- `scripts/multidomain/check_shared_memory.py`：16 GiB 启动空闲下限、64 MiB 实际写读探针；只清理自身探针文件。

## 验证范围与比较口径

发布前本地进行 CPU 单元测试、候选池重选集成测试、固定版本 verl Hydra 配置编译、Git worktree 安装测试。GPU smoke、反向容量、正式训练需要在 ABCI 执行。

四域数据规模预计大于六域均衡训练集。同为 3 epochs 时，预计完整 batch 数不同，因此后续比较应同时报告训练步数、采样输出数和训练 tokens；不能把差异全部归因于域组合。训练集整体 25% 的比例不保证每个随机 batch 精确均衡。
