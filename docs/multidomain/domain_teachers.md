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

首次 SSH 连接使用 `StrictHostKeyChecking=accept-new`：将首次见到的节点密钥记录到正常的 known_hosts，之后仍拒绝密钥不匹配。原来的 `BatchMode=yes` 配合默认 `ask` 无法进行首次确认，会直接报 `Host key verification failed`。单凭这行错误不能断定是首次连接还是密钥冲突；此修复只处理首次记录，不删除或覆盖已有信任记录，也不关闭主机密钥检查。该选项仅用于旧 guardian 的一次性 SSH 交接，不修改全局 SSH 配置。若仍报密钥变化或 public-key 认证失败，应依据具体错误核实节点身份或认证条件；此时不会提交替代 job。

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

## Math step 50、CP step 80 和 base 的补充 test

更新 checkout 后，在任一登录节点执行：

```bash
"$teacher_python" -I scripts/domain_teachers.py evaluate-checkpoints --campaign "$teacher_campaign"
qstat -u "$USER"
```

**这条命令会交接/启动监护并实际提交 GPU 作业。** 固定评估 Math 的完整 checkpoint 50、CP 的完整 checkpoint 80，以及冻结的未经训练 base。三个目标都使用同一份四域 test（本 campaign 共 994 条）、greedy n=1 和当前已安装的 final-test Math 评分器修复。test 不更新训练权重，也不改变原 validation best 或 early stopping。

| 任务名（status 中显示） | 执行顺序 | 结果位置（campaign 下） |
| --- | --- | --- |
| `math_step50` | 独立 HF 导出 job → 独立四域 test job | `runs/math_step50/test.json` |
| `cp_step80` | 独立 HF 导出 job → 独立四域 test job | `runs/cp_step80/test.json` |
| `base` | 直接四域 test job | `runs/base/test.json` |

每个 allocation 申请 1 节点、8 GPU、24 小时 walltime；无中断时这三个目标合计 5 个 allocation，两个导出结束后才各自提交 test。它们由现有 campaign guardian 一起监护，与 Logic 共用原并发上限（默认最多 4 个 job）。base 不再等待四个 teacher 全部导出；已排队/运行/完成的 base 会被接管，不重复提交。已有 Logic GPU 作业继续受监护，交接只替换登录节点上的控制进程。新版 guardian 的共享文件交接不要求用户登录原节点。

命令先核对 Math 50、CP 80 的训练完成记录、完整 checkpoint marker、lineage 和文件 inventory。缺少 checkpoint 时在交接监护前报错，不会静默改评其他步数。安装时在独占控制锁和原 run writer 锁下硬链接所需模型分片，然后原子发布 `checkpoint_evaluation.json`。独立运行目录中的 `best.json` 只是复用导出接口的指定权重描述，明确标记 `selection_source=explicit_checkpoint`、`score=null`，不伪造 validation 选择。原 `runs/math`、`runs/conversational_pivot` 的 best/export/test 和完整训练 checkpoint 保留。

重执行同一命令会验证已注册目标并接管原任务，不清除 job ID、pending intent 或 journal，不重做已完成任务。新 job 的运行源码按哈希冻结在 `checkpoint_evaluation_runtimes/`，已有 template/训练合同不修改。导出后验证全部 HF 权重加载及 finite forward，再释放额外的原生模型硬链接。

监护沿用已确认 PBS 271 和 walltime 超限的自动续投条件；没有“连续三次无进度就停止”的上限。test 每完成一批便持久化，恢复时只处理剩余批次；生成到一半、未提交的 batch 需要重做。未完成的 HF 导出可以重跑；已发布且有加载证明的 HF export 直接复用。PBS 查询失败继续观察原 job，不据此重复提交；应用错误仍保留为 `BLOCKED_APPLICATION`，显式 STOP 优先。监护持续至所有需处理任务完成、停止或确认阻塞，24 小时仅是单个 GPU allocation 的 walltime，不是整个监护时长。

可用原 `status` 查看新增 task 的阶段、job 和心跳；`stop/reenable --domain math_step50` 或 `cp_step80` 只操作该补充任务，campaign 级 stop 覆盖它们。原 `teachers.json` 继续表示 validation 选出的模型；补充测试结果单独保存于上表路径。此次命令不导出或诊断训练日志。

### 单次恢复启动阶段被 SIGKILL 的补充导出

补充导出若在写出 `result.json` 前被 PBS 以 `Exit_status=265` 终止，控制器会保留为 `BLOCKED_APPLICATION`。这个名称包含未获自动重试许可的终止，不证明 Python 应用已经报错。265 不在原自动 271/walltime 策略内。

已调查该次中断并决定单次重试时，在任意登录节点执行：

```bash
"$teacher_python" -I scripts/retry_teacher_export.py --campaign "$teacher_campaign" --task math_step50 --expected-job YOUR_JOB_ID.pbs1
```

**此命令会实际安排一次新的 GPU 导出，成功后自动进入 test，并继续由原 campaign 监护。** 导出和 test 各申请 1 节点、8 GPU、24 小时 walltime；后续确认的271/walltime中断沿用原自动恢复。它只接受已经注册的补充 checkpoint 任务、指定旧 job、明确的265终态及尚未写出 `result.json`、`export.json`、`test.json` 的情况。它会重新查询 PBS 并核验所属用户、队列、名称、提交路径、冻结 request、checkpoint 和评分器身份；未知查询、活跃作业、显式 STOP 或 writer 锁冲突均不会清除原作业所有权。

旧状态与 PBS 观察先保存到 `repairs/checkpoint-export-sigkill-once-v1/`，随后在 guardian/control/writer 锁下将该导出设为 READY。原作业目录、模型、已完成的 CP/base 结果及 Logic 作业保留。重复执行同一个 `--expected-job` 会接管已恢复的流程，不会再次投递；若新作业再次以265结束，需要另行调查。本入口不把所有265或应用异常加入无限自动重试。

本地验证包含单次提交、export→test、后续271恢复、再次265时保持阻塞、重复运行与不明 qsub 所有权、PBS身份和状态检查、STOP/writer锁、checkpoint改动、审计写入中断恢复，以及真实 guardian 子进程交接并接管 Logic。PBS命令和模型权重为合成fixture；实际GPU提交与导出结果需以部署后的job ID和收据为准。

## DeepMath 补充 held-out：base、Math20、Math50

本 campaign 的 Math 训练包含 Nemotron 和 DeepMath L8/L9，而原 Math validation/test 只含 Nemotron。新增入口固定比较未经训练的 base、原 validation 选出的 Math step 20、训练末尾 Math step 50，分别报告 DeepMath validation/test 及 L8/L9 分数。此比较不改写原 best、early stopping 或原四域 test；结果属于补充诊断，不能把看过结果后挑选的模型再描述为一次未参与选择的最终 test。

在任一登录节点更新 checkout 后执行（`run` 包含准备和实际投递）：

```bash
teacher_python=/groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d/.venv-multidomain/bin/python
teacher_campaign="$PWD/outputs/domain_teachers/qwen30b_d4_teachers_seed42_v1"
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py run --campaign "$teacher_campaign"
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py status --campaign "$teacher_campaign"
qstat -u "$USER"
```

`prepare` 只做 CPU 数据准备，加载 tokenizer 和隔离评分器，不加载模型权重、不提交 GPU；`start` 只注册已准备的评估并交接/启动 guardian；`run` 顺序执行两者。CPU 参考答案检查期间每 32 个候选题打印进度，并保存可复用的参考答案检查缓存。三个评估任务各申请 **1 节点、8 GPU、24 小时 walltime**，每个任务在同一 allocation 中按 validation→test 顺序完成两份互斥数据，分别汇总分数。无中断时合计三个新 allocation，复用已有 HF 导出，不重新训练或导出 Math。

| status 任务名 | PBS job 名 | 权重来源 | campaign 下的最终收据 |
| --- | --- | --- | --- |
| `deepmath_base` | `0390_dm_base` | 冻结的原始 HF base | `runs/deepmath_base/heldout.json` |
| `deepmath_math20` | `0390_dm_m20` | `runs/math/export.json`，必须是 step 20 | `runs/deepmath_math20/heldout.json` |
| `deepmath_math50` | `0390_dm_m50` | `runs/math_step50/export.json`，必须是 step 50 | `runs/deepmath_math50/heldout.json` |

Math50 导出未完成时，DeepMath 对应任务保持 `WAITING_MODEL`（并发已满时可能尚为 READY），待已有导出发布且验证通过后提交；不会并行发起第二个导出。既有 Math50 四域 test、Logic 训练和三个新任务共用同一个 guardian、同一个最多四个 job 的限制。原任务和已完成结果保留。重复 `run/start` 接管已注册任务，不清空 job/pending、不重复提交。

数据准备固定 seed 42，默认每份 256 题。优先复用找到的旧 DeepMath holdout，再以未进入本 campaign 训练/评估的数据补足；旧文件有明确 validation/test 名称时保持其分区。L8/L9 配额依据冻结 Math 训练中两类 DeepMath 的比例，按当前 6268:1270 得到每份 213/43。两份数据严格互斥。所有候选题均排除本 campaign **全部四域 train/validation/test** 的题目，以及发现的旧 probe；比较规范化题文并处理训练提示词外壳，跨来源重复和答案/难度冲突也剔除。这是明确规则下的文本去重，不声称完成语义近重复或模型预训练污染审计。

读取旧 holdout 的题目/答案/难度，**不沿用它的 Unicode prompt**。准备程序从实际冻结 DeepMath 训练行和 raw 题目反推唯一 prompt 外壳，核验原行 token hash；新题使用相同 system/user scaffold、tokenizer、无工具设置、64K 总上下文及相同每条输出预算规则。三模型使用同一冻结数据、greedy n=1 和当前安装的 final-test Math verifier。参考答案必须通过该评分器的 CPU 自检；图像题、超长输入和无法自洽评分的参考答案不入选。评分器基础设施异常直接报错，不当作零分或正常数据过滤。

入口仅在已知项目 data/assets/supplemental_data 目录内查找输入，不下载数据。输入发现和显式补充方式：

```bash
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py discover --campaign "$teacher_campaign"
# 如自动发现不到 raw 输入，可显式提供；三个路径参数均可重复。
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py prepare --campaign "$teacher_campaign" --input /path/to/deepmath_l89_input.jsonl.gz --holdout /path/to/deepmath_holdout.parquet --exclude-probe /path/to/deepmath_probe.parquet
```

必须有 raw 题目用于核对训练 prompt。若清理后无法凑足指定 L8/L9 配额，准备会在 GPU 投递前明确报出缺口，不拿 L8 偷补 L9 或重用训练题；首次准备可用 `--size` 显式缩小每份大小。已经发布的数据不可用新参数覆盖。数据、输入 SHA256、实际来源计数、去重审计、模板证据和自检身份冻结在 `deepmath_evaluation/data/manifest.json`。任务及模型选择冻结在 `deepmath_evaluation.json`，新评估源码另存于不可变 runtime 副本；原训练 template 不修改。

监护延续到任务完成、显式停止或确认应用阻塞；**24 小时只是单次 GPU allocation 上限**。确认 PBS 271 或明确 walltime 超限后自动续投，无连续三次上限。完成的 batch 存在 `evaluations/heldout_*/`，模型/数据/评分器/依赖/生成参数/顺序完全一致才可复用；未完成的 batch 重做。PBS 查询不明继续观察原 job，应用错误和 265 不会被自动泛化为无限重试。已有共享文件锁交接支持从任意登录节点执行命令。`stop --domain deepmath_base` 等入口和 campaign 级 STOP 同样生效。

查看进度、生成比较报告：

```bash
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py status --campaign "$teacher_campaign"
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py report --campaign "$teacher_campaign"
```

报告保存在 `deepmath_evaluation/comparison.json`，包含每模型每 split 的整体/L8/L9 分数，及同题配对的分数差、改善/退步题数、10000 次 paired bootstrap 的 95% 区间（seed 42，探索性未校正区间）。报告从有完整校验的逐题 journal 重算分数；不同依赖/生成协议不混合。尚未完成时输出 `INCOMPLETE`，不把部分批次当完整 test。原 `teachers.json` 仍表示原 validation 选出的模型。

本地 CPU 测试覆盖真实 Parquet 输入、旧 Unicode holdout 解析、训练和 probe 去重、分层互斥、模型及数据绑定、冻结 launcher、Math50 等待、271 多次续投、未知 PBS 所有权、注册中断、实际 pinned validation 循环跨 split 恢复，以及真实 guardian 子进程交接。GPU/模型权重与 PBS 调度为 fixture；实际集群运行以返回的 job ID 和完成收据为准。

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

## 一次查阅所有训练与补充评估

`scripts/inspect_domain_teachers.py` 是只依赖标准库的只读报告入口，支持从 `git show REV:scripts/inspect_domain_teachers.py | python -I -` 执行，无需更新正在使用的 checkout 或替换 guardian。

```bash
"$teacher_python" -I scripts/inspect_domain_teachers.py --campaign "$teacher_campaign" --logs 12
```

输出涵盖监护锁/心跳、四个 teacher 的最近记录步与完整保存步、validation best/历史、最近五条训练 reward/零 advantage 组比例/响应长度/耗时、四域 test 矩阵（原 teacher、base、Math50、CP80）、DeepMath 两份 held-out 的整体和 L8/L9 分数，以及所有任务的排队/运行/阻塞、当前或上次 job 和实时 PBS exit/walltime。未注册的补充任务明确显示未注册，已准备数据不会被解释成已提交作业。

`--logs N` 有界读取最新 job 的日志末尾，默认不读日志；`--history N` 控制近期训练行数；`--no-pbs` 仅读持久化文件。PBS 每个查询超时20秒、最多3个并行，超时/缺失记录显示 UNKNOWN。报告不调用 qsub/qdel，不启动或停止监护，不写 campaign 状态；可由 shell 的 `tee` 保存成小型文本文件。结果是非原子的运行快照，收据摘要校验不代替重新校验模型和完整生成 journal；训练重复步仍保留以显示中断重跑，旧 error 字段不等同于当前失败。

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

2026-10-30 首次 SSH 主机信任修复验证：6 项监护交接回归通过；另用真实 OpenSSH 客户端和仅监听回环地址的临时 SSH 协议服务器验证四种情况：旧 BatchMode 调用拒绝未知主机、修复后首次连接记录密钥并成功、重复连接成功、同一地址更换主机密钥时拒绝远端命令且保留原 known_hosts。临时服务器使用内存中生成的测试密钥，不执行请求中的命令；不修改系统 SSH 配置，生产无需新增 Python 依赖。此验证覆盖主机密钥处理，不证明 ABCI 实际账户认证已成功。

2026-10-31 指定 checkpoint 补充评估验证：76 项 teacher 测试（含新增 13 项）与 167 项 continuation 回归全部通过，共 243 项测试及 29 个 subtest；Ruff E4/E7/E9/F、格式与 diff 检查通过。覆盖 Math 50/CP 80 的独立模型硬链接和导出请求、原 best/test 保留、缺失 checkpoint 提前失败、安装中断恢复、base 提前且不重复提交、与 Logic 共用 4-job 上限、导出和 test 的多次 271/walltime 续投、journal 保留、未知提交状态、评分器版本隔离及任务 stop。真实隔离 guardian 子进程验证共享文件交接后接管原 Logic job，并验证重复命令不再提交或取消作业。PBS 和权重使用本地 fixture；GPU 导出及 test 尚需在 ABCI 部署执行。
