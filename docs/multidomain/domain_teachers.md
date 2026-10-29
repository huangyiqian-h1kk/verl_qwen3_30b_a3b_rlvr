# 四个独立 domain teacher

入口为 `scripts/domain_teachers.py`，配置为 `config/domain_teachers/campaign.json`。这是新 campaign，每个 teacher 从同一份 Qwen3-30B-A3B-Instruct-2507 HF base 开始，使用自己的训练、评估、checkpoint、停止状态和 PBS job。

## 固定实验协议

| 项目 | 实现 |
| --- | --- |
| teacher | `math`、`if`、`conversational_pivot`、`logic_algorithmic` |
| 数据 | 从已冻结 `four_domains_10k_v1` 按 domain 直接过滤；每域 train 10,000 |
| validation | Math 238，其余各 256；本域完整集合，greedy、n=1 |
| test | 保留父数据实际 ID、split membership 和数量；最终才使用 |
| 训练 | 同步 GRPO；256 prompts × 8 responses；minibatch 32；每步 8 个 optimizer minibatches |
| GPU | 每 teacher 一物理节点、8 GPU；actor TP2/PP2/EP2/CP1；rollout TP4、2 replicas |
| 参数 | LR 1e-6 constant、clip 0.2、token-mean、temperature/top-p 1、seed 42；KL 和 entropy bonus 为 0 |
| 内存 | 参数/optimizer/gradient offload、gradient checkpointing；共享 context 65,536 |
| 生成预算 | `min(generation_max_tokens, 65536 - prompt_tokens)`；沿用冻结 tokenization hash |
| 步数 | 最多 200 GRPO steps；显式构造 200 个完整 batch，不受旧 3-epoch 长度限制 |
| 保存 | 每 5 步同步完整保存，最终步也保存 |
| 评估 | step 0、每 10 步、结束时；无按耗时调整到 15 步的逻辑 |
| 选择 | 本域 validation 平均 verifier score；相同分数保留较早 checkpoint |
| early stop | 最早 step 50；距最近 best 至少 25 training steps，且只在完整 validation 后决定 |
| PBS | `gcg51557` / `R9920261000` / `rt_HF` / 24 小时；不指定主机 |

因为 validation 在 10-step 网格上，best 在 step 40 且以后不再改善时，step 60 尚未达到 patience 25，step 70 才停止。step 0 也参与 best 选择；若训练始终没有超过 base，可以导出并验证原始 base。

采样使用固定 seed 的打乱池：每次抽 256 个不重复 prompt，池尾不足一个 batch 时丢弃短尾并重新打乱。每域每轮 39 个 batch（9,984 prompts）；200 步共 51,200 prompt exposures。每个完整 checkpoint 保存相同计划的 identity 和真实 StatefulDataLoader cursor。跨重启恢复模型、optimizer、LR、RNG 和 cursor；这不承诺 vLLM 跨进程的采样结果逐 token 一致。

## 在 ABCI 安装与启动

以下是待在 ABCI 执行的命令。本地 CPU 测试不代表已经在 ABCI 提交或完成 GPU 训练。

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR
git clone --single-branch --branch codex/d4-domain-teachers \
  https://github.com/huangyiqian-h1kk/verl_qwen3_30b_a3b_rlvr.git \
  data_mixture_rl_4d_teachers
cd data_mixture_rl_4d_teachers

module load gcc/13.2.0 cuda/12.8/12.8.1 cudnn/9.10/9.10.2 nccl/2.29/2.29.7-1
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate /groups/gcg51557/experiments/0390_rlsd/envs/verl_qwen3_moe_megatron_py312_cu128

teacher_parent=/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d
teacher_python="$teacher_parent/.venv-multidomain/bin/python"
teacher_campaign="$PWD/outputs/domain_teachers/qwen30b_d4_teachers_seed42_v1"

"$teacher_python" -I scripts/domain_teachers.py prepare --campaign "$teacher_campaign"
"$teacher_python" -I scripts/domain_teachers.py preflight --campaign "$teacher_campaign"
"$teacher_python" -I scripts/domain_teachers.py start --campaign "$teacher_campaign" --background
"$teacher_python" -I scripts/domain_teachers.py status --campaign "$teacher_campaign"
```

无需安装依赖。默认最多并发 4 个本 campaign 的 job；如需 1–3 个，在第一次 `prepare` 加 `--max-concurrent 2` 等。已准备 campaign 的协议不可原地修改。并发上限只计算此 campaign 自己的 job/未决提交，不对整个账号限流。

`prepare` 校验父数据 manifest，并保存所有分片的 SHA256、sample IDs 和来源/类别/task 计数；逐字段检查 Parquet round trip。不会重新采样 train/validation/test 成员，也不会重新制作补充数据。默认来源是实际运行过的 `formal_4f2e239caeda4a178315ef9ad0899af0/request.json`；按该 request 的哈希复制部署代码、Math isolated backend zip、历史 reasoning-gym verifier 和完整状态 audit。归档里未包含的二进制和 audit source 直接从 ABCI 原 job 读取；缺失或变化会在提交前报错。

CPU `preflight` 检查 pinned verl、数据、tokenizer、模型资产 inventory、真实 Math verifier 子进程及真实 torchdata 保存/恢复，并编译配置。正式 allocation 再检查一节点八 GPU、NCCL 和运行环境，然后直接开始正式训练；没有额外 pilot training。基础环境和模型路径继续使用原安装。

## 恢复、停止与监护

```bash
# 登录会话退出后 guardian 继续运行；若 guardian 已退出，重新接管已有 job。
"$teacher_python" -I scripts/domain_teachers.py resume --campaign "$teacher_campaign" --background

# 停止一个 teacher；省略 --domain 则停止整个 campaign。
"$teacher_python" -I scripts/domain_teachers.py stop --campaign "$teacher_campaign" --domain math
"$teacher_python" -I scripts/domain_teachers.py status --campaign "$teacher_campaign"

# 明确确认已停止的 teacher 可以再次运行。
"$teacher_python" -I scripts/domain_teachers.py reenable --campaign "$teacher_campaign" --domain math
"$teacher_python" -I scripts/domain_teachers.py resume --campaign "$teacher_campaign" --background
```

`stop` 先写 durable fence，再取消已验证 owner/queue/name/提交路径的准确 job。已在途的 qsub 会被找回并取消。`STOPPING` 表示尚未观察到 PBS terminal；若 guardian 当时未运行，重复 `stop` 可重新观察。必须确认 terminal 才能 `reenable`。普通 `start/resume` 无法清除 stop fence。对全 campaign 的 fence，需要不带 `--domain` 的 `reenable`。

监护无固定截止时间，使用共享文件系统 flock；heartbeat 显示主机、时间及 stale 状态，不按另一主机上的 PID 杀进程。Q/R job 被接管，H 保持原 job 并显示 `BLOCKED_HELD`。qstat 超时、查询不明或 qsub 结果不明时显示 `BLOCKED_UNKNOWN`，保留 intent/job，不再盲投。`BLOCKED_UNKNOWN` 和 `BLOCKED_HELD` 都不是监护退出条件：即使其他 teacher 已完成或发生应用错误，监护仍按原轮询间隔自动重查、更新心跳，PBS 恢复后接着处理原作业，不需要手动 `resume`。仅当监护进程本身已退出时才需要重新启动。所有任务均完成、显式停止或确认需要人工处理的错误时，监护可以退出；等待缺失导出的 base test 不会令它空转。

只有已观察到 walltime exceeded（`-29` 且对应 comment）或管理员终止（`271`）才自动续投；应用错误会阻塞该 teacher。沿用旧 4-domain 正式训练的规则：即使中断前没有新增完整 checkpoint/已完成 eval rows，符合这两种原因仍会续投，没有“三次无进度即停止”的上限。`no_progress_retries` 只作为诊断计数，有进展会清零；旧 campaign 合同中冻结的 `max_no_progress_retries` 字段不再作为续投限制，无需修改合同。一个 teacher 被阻塞时，其他 teacher 继续。后台进程退出或异常请查看 `guardian.log` 和 `status`。

### 修复已经启动的旧监护器

初始 teacher 实现额外加入的三次上限偏离了旧 4-domain 正式训练规则。第一版修复还把 `BLOCKED_UNKNOWN`/`BLOCKED_HELD` 作为监护终态，导致其他 teacher 已结束时，一轮 PBS 查询失败就可能使监护退出。当前修复同时覆盖这两个问题。更新 Git checkout 本身不会更新 campaign 内冻结的监护器；已经安装过第一版修复也需要再次执行以下命令，以发布新的监护副本。命令可在任一登录节点执行，不需要根据 heartbeat 手动切换节点：

```bash
"$teacher_python" -I scripts/domain_teachers.py repair-retries --campaign "$teacher_campaign"
"$teacher_python" -I scripts/domain_teachers.py status --campaign "$teacher_campaign"
```

该命令先验证冻结合同和 CPU preflight，在独立 `controller_revisions/` 下生成并校验监护代码副本；取得 launch/control 锁后，在旧监护器不处于 PBS 事务中的时刻交接 guardian 锁。它不调用 `qdel`，不修改训练数据、冻结 template、campaign identity、已有 job request、checkpoint 或评估 journal。

新版 guardian 支持 `shared-filesystem-guardian-handoff-v1`：修复端在共享 campaign 目录写入绑定准确进程实例的短期请求，并持有该请求专用的文件锁；旧 guardian 在等待下一次轮询或等待 control 锁时响应请求、正常退出，新 guardian 在当前节点接管。请求过期、请求端退出、进程实例变化或请求被撤销时不会触发交接。跨节点交接依赖共享文件锁，不依赖节点间 SSH，也不要求用户记住运行主机。

已经运行的旧代码尚不支持共享请求，首次迁移会自动使用有超时的非交互 SSH 连接记录中的 `loginN`，在该节点验证用户、准确命令行和 pidfd 后停止旧监护器，再由当前节点取得共享锁。SSH 保留正常主机密钥校验；不把远端 PID 当成本机 PID 使用。节点间 SSH 若因认证、主机密钥或网络问题不可用，会报告具体错误；只要原锁仍被持有，就不会启动第二个 guardian，也不会解除 CP 阻塞或新增投递。已完成远端停止但回执丢失时，以实际共享锁是否释放为准。交接证据保存在 `repairs/guardian_handoffs/`；正常交接不改变已提交 GPU job。新版 guardian 的 `guardian_runtime.json` 包含 `handoff_policy` 和每次启动唯一的 `incarnation`。

修复会为每个 `BLOCKED_RETRIES` 重新查询其 `previous_job`，校验 owner/queue/name/提交路径以及 campaign/domain/phase，只有确认 271 或 walltime -29、且没有 job/pending 或显式 stop fence 时才改回 READY。仍持有 job/pending 的 `BLOCKED_UNKNOWN` 由新监护正常重查：确认 271/walltime 后续投，确认 Q/R 则接管原 job，确认应用错误则保持应用阻塞；未知状态本身不授予续投权限。显式停止始终优先。READY 的任务由新监护器按原并发上限续投，从最新完整 checkpoint 恢复；日志中尚未保存的步数不作为恢复点。修复之前的 state/PBS 证据保存在 `repairs/four-domain-retries-v1/`。

`REPAIRED_GUARDIAN_RUNNING` 表示已确认新监护代码持锁且完成第一次 tick；旧重试上限的恢复另看 `recovered` 是否为 `REARMED`，以及对应 `states` 中的新 job ID。对于持有 job 的 `BLOCKED_UNKNOWN`，`recovered` 可以为空，应检查 `states` 的实际状态和 job ID，不能把监护正常等同于所有 teacher 都正常。`REPAIR_INSTALLED_CHECK_GUARDIAN` 表示尚未获得第一次 tick 的确认，PBS 连续超时可能使首次 tick 超过命令的 20 秒等待窗口，应检查后续 status/guardian.log。心跳中的 `retry_policy` 应为 `four-domain-external-interruption-v1`，`observation_policy` 应为 `persistent-pbs-observation-v1`。重复执行同一修复不会重新提交已有 job，也不会重启已验证的新监护器。启动入口会持续指向校验过的修复副本，旧的 start/resume 入口也不能重新启动旧策略。

已有本域训练完成但四域 test 出错的 teacher 不会被此命令解除应用错误阻塞；应保留导出模型，收集 `collect` 诊断包定位 verifier 错误后再恢复评估。

### 修复 CP 最终 test 的 And/Equality 评分器错误

已确认的错误是 `AttributeError: 'Equality' object has no attribute '_unsorted_args'`：冻结的 `latex2sympy2_extended==1.11.0` 在构造 `And` 时，无条件给返回值写入参数顺序元数据；SymPy 可能把该表达式化简成 `Equality` 或布尔值，导致比较过程异常。修复只在返回值仍为该 `And` 类型时设置元数据，保留原化简和数学比较，不把基础设施异常直接记为零分。

更新 checkout 后，可在当前任一登录节点执行，命令会自动完成上述监护交接。`teacher_diagnostic` 指向失败 test 作业的原始 `math_diagnostics/*.json` 或其完整副本：

```bash
"$teacher_python" -I scripts/domain_teachers.py repair-math-test --campaign "$teacher_campaign" --domain conversational_pivot --diagnostic "$teacher_diagnostic"
"$teacher_python" -I scripts/domain_teachers.py status --campaign "$teacher_campaign"
qstat -u "$USER"
```

该命令会执行部署并恢复投递，不只是诊断。先确认 frozen request、原始诊断哈希、Math test sample ID 和 PBS 已结束且 exit 1，再用本 campaign 原始 pinned ZIP，在 CPU 隔离子进程中重现错误、验证修复后的正常评分，运行原有 13 项 Math selfcheck 和 4 项正负比较检查。检查失败时不停止 guardian、不解除 CP 阻塞、不提交新 job；不需要修改共享 conda 环境或安装依赖。

CPU 检查通过后，命令使用与 `repair-retries` 相同的锁和主机/PID 校验交接监护器，同时包含此前的持续观察和 271/walltime 续投修复。取得独占权后再次确认原失败作业，发布 `test_verifier_revision.json`，仅把该 teacher 的 **test** 阶段改回 READY，由 guardian 按原并发上限提交。已导出的 best HF 模型直接复用；CP 选择 step 50，训练完成步数仍为 80。不会重新训练、重新导出、清除 stop fence 或取消已有 GPU 作业。Logic 的已有作业由 guardian 正常观察，只有确认 271/walltime 才按既定规则续投。

修复代码存放在独立的 `test_verifier_revisions/`，原 template、campaign identity、训练/validation 评分器和已有 job request 保持原样。补丁只注入未来 **final test** 的隔离 Math worker；CP、后续 Logic/base test 和这些 test 的 271/walltime 续投都使用此版本。新 verifier identity 使 CP 新建完整的 994-row test journal，旧 216 条已完成记录保留但不复用，避免新旧评分混用。Math/IF 已完成的旧 test 不会自动重跑。

`MATH_TEST_REPAIR_INSTALLED` 表示补丁已安装，另看 `guardian.status`、实时 status 和新 CP test job ID 判断投递/排队/运行；它不代表 test 已完成。命令返回 `ALREADY_INSTALLED` 时不会再次解除状态，也不会重复提交已拥有的 job。`REPAIR_INSTALLED_CHECK_GUARDIAN` 表示 20 秒内尚未确认新监护器第一次 tick，应继续检查心跳和日志。CPU preflight 保存在 revision 目录，交接证据保存在 `repairs/math-test-and-typeguard-v1/`。其他应用错误仍按原规则阻塞，需要诊断。

`teachers.json` 增加 `test_verifier_identities`、`expected_test_verifier`、`tests_use_current_verifier` 和 `comparable_tests_complete`。`tests_complete` 仍只表示四个 teacher 加 base 的结果均已完成；当 Math/IF 保留旧评分器结果时，不能把它当作统一评分版本的最终矩阵，`comparable_tests_complete` 为 false。若后续需要统一比较，应使用新版本显式重评旧结果。

## 评估恢复、导出和保存空间

validation/test 的每个完成 batch 都原子写入 journal，绑定 model/checkpoint、数据、verifier、依赖版本、生成参数和顺序。重启先恢复当前 checkpoint 上未完成的 validation，再进入下一个训练 update；已完成 batch 不重新生成。收据核对完整 ID 集和输出哈希。validation 的 attempted/completed 耗时保存在 `evaluations/*/attempts`；训练及保存耗时保存在 `jobs/*/train/step_*.json`。不会依据这些耗时修改 eval interval。

完整保存先写 staging，验证模型/optimizer/LR/RNG/dataloader、8 rank audit 后提交 marker。训练中保留 latest 2 full，加一个 best 模型权重硬链接；正在评估的当前 checkpoint 在任何下一步保存前受保护。停止训练后保留 latest 1 full。

默认自动衔接独立的 export job 和 test job，也计入 campaign 的 4-job 上限：

1. 从 validation best 的原生权重导出 HF；export-only 进程不恢复 optimizer、不进行训练 update。
2. 前一 Ray 进程释放 GPU 后，用新的 Python 进程加载全部 HF shards，检查 missing/unexpected keys、tokenizer/chat-template identity，并执行 finite forward。
3. 完成加载验证后发布 `export.json`，释放 best 原生权重硬链接；最终保留 latest 1 full 和已验证的 best HF export。若 base 获胜，导出指向冻结 base 资产并附加载证明。
4. 用每个选定 teacher 在四域冻结 test 上 greedy n=1 评估；base 额外跑一遍同样四域 test。test 不参与 early stop 或 best 选择。
5. 写 `teachers.json`，包含四个 teacher 的路径、best step、validation score、权重/verifier identity、load proof、4×4 test matrix 和 base test。`tests_complete` 表示全部 test 已完成。

只有已提交的 full checkpoint 才能恢复。新 writer 会清理前一中断产生的 staging、未发布临时 hardlink 和 export candidate，不删除已提交 checkpoint 或已发布 HF artifact。大权重 payload 的完整性检查采用 size/mtime inventory，小文件额外 SHA256；不把它描述为全部 tensor 的逐字节相等证明。

手动重跑后处理、检查实际 allocated blocks 或收集诊断：

```bash
"$teacher_python" -I scripts/domain_teachers.py export-best --campaign "$teacher_campaign" --domain math
"$teacher_python" -I scripts/domain_teachers.py evaluate --campaign "$teacher_campaign" --domain math
"$teacher_python" -I scripts/domain_teachers.py storage --campaign "$teacher_campaign"
"$teacher_python" -I scripts/domain_teachers.py collect --campaign "$teacher_campaign"
```

`storage` 按 inode 去重硬链接；`collect` 保留配置、日志、rank/restore 证据和评估 journal，排除模型/optimizer 大文件、Parquet 和 verifier zip。每个 teacher 的 `run.json`/`state.json`、`progress.json`、`validation_state.json`、`best.json`、`training_complete.json` 和 `export.json` 是主要检查入口。

## 本地验证

```bash
python -m unittest discover -s tests/multidomain -p 'test_domain_teachers*.py' -v
python -m unittest discover -s tests/multidomain -p 'test_training_continuation*.py' -q
```

新增测试需要 PyArrow、NumPy、PyTorch 和 torchdata。Math 修复的本地测试另需 `math-verify==0.9.0`、`latex2sympy2_extended==1.11.0`、`sympy==1.14.0`、`mpmath==1.3.0` 和 `antlr4-python3-runtime==4.13.2`，测试会将这五个版本打包成独立 ZIP，并与生产使用的原始 ZIP 哈希区分；ABCI 部署直接使用冻结 ZIP，无需安装这些包。覆盖实际 Parquet 分片、实际 torchdata cursor、固定 pinned validator 的中断恢复、模型选择与 hardlink retention、PBS intent/stop/held/unknown/retry/concurrency 故障。GPU Megatron 保存/恢复、HF 导出和完整训练仍需在 ABCI 正式执行时取得证据。

2026-10-28 续投修复验证：42 项 teacher 测试（含新增 15 项修复测试）和 167 项 continuation 回归全部通过。修复测试运行真实隔离 Python 子进程、pidfd/SIGTERM 和文件锁，验证旧监护接替、缺少 Python pidfd wrapper 时的内核兼容路径、错误主机/命令行拒绝、停止优先、旧 start/resume 入口、发布中断恢复、准确一次续投，以及训练合同/数据/checkpoint/已有 request 不变。PBS 命令和权重 payload 使用本地 fixture；这些结果不代表已经在 ABCI 部署或恢复 GPU 训练。

2026-10-29 持续监护修复验证：49 项 teacher 测试和 167 项 continuation 回归全部通过，Ruff E4/E7/E9/F、格式及 diff 检查通过。新增测试覆盖只有最后一个 teacher 未完成时的连续查询失败、H→Q→R、不明 qsub intent 的原作业找回、未知状态下停止优先，以及确认应用错误后的正常退出。真实隔离子进程先运行上一版 controller，复现查询失败导致其正常退出；随后升级冻结监护副本，验证 271 只续投一次、下一轮连续 PBS 故障期间锁和心跳保持、查询恢复后接管原作业，以及冻结合同/数据/checkpoint/旧 request 不变。测试使用模拟 PBS 命令，并不代表本补丁已经部署到 ABCI。

2026-10-29 Math final-test 修复验证：57 项 teacher 测试和 167 项 continuation 回归全部通过，Ruff E4/E7/E9/F、格式及 diff 检查通过。新增 8 项测试覆盖真实隔离评分子进程中的原错误复现、正确/错误答案及原 13 项 selfcheck、preflight 失败不改变作业、监护交接、重复修复不重复投递、271 后保留修复评分版本、不同 verifier 缓存隔离、stop/作业身份限制和混合评分版本的汇总标记。另以用户提供的完整失败输入在相同五个依赖版本的隔离子进程重放：原版报同一异常，修复版正常比较并返回 0 分。公开 fixture 仅包含通用合成输入和原评分源码，不包含该完整诊断或模型输出。生产部署会在 ABCI 使用原始冻结 ZIP 再次执行 preflight；本地结果不代表已提交或完成 GPU test。

2026-10-30 跨登录节点交接验证：63 项 teacher 测试和 167 项 continuation 回归全部通过，Ruff E4/E7/E9/F、格式及 diff 检查通过。使用真实旧版 controller 子进程、文件锁、pidfd 和隔离远端 helper，验证从其他节点自动迁移旧 guardian、新版共享请求交接、请求端退出/过期/错误实例拒绝、SSH 失败不改变原 owner/job、SSH 回执丢失后以共享锁为准，以及重复修复不重复投递。CP 修复集成测试也通过同一远端 helper 完成 CPU preflight、交接和 test 恢复，并验证 271 续投保留评分器版本。测试中的 SSH/PBS 传输和主机名使用 fixture；尚未在 ABCI 验证实际节点间 SSH 连通性或提交新 GPU job。
