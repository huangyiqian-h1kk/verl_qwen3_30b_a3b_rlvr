# MOPD 单双节点执行与完整状态交接

本版在同一个冻结科学 manifest 下增加 `single_node` 与 `dual_node` 两种执行配置。
单节点将 student 生成、四 teacher 评分、student 学习分阶段放在同一节点的 8 张 H200 上。
双节点保留 student 8 卡与 teacher 8 卡分离的原生流水方式。

## 保持的实验设置

- B=2048、R=1、原始逐样本生成预算、64K 总上下文、正式 200 次更新。
- student 始终 world8，TP2/PP2/EP2；rollout 为两组 TP4；teacher 始终四组 TP2。
- 原有微批、65536 token/GPU、max_num_seqs=16、max_num_batched_tokens=8192、BF16 均保持。
- 原生 sampled-token k1 PG、detach/clip5、gpg、seq-mean-token-mean；不引入任务奖励。
- 单节点评分阶段两臂均唤醒全部四个 teacher，Only 中未选中的 teacher 保持空闲。

单节点先完成整批生成、确认 student rollout 休眠，再唤醒四个 teacher。评分使用原始
token IDs、域与样本顺序；原生 next-token 对齐和 padding 保留。所有评分完成并确认四个
teacher 休眠后，才进入 student old-logprob、梯度计算和一次 optimizer update。
每个 CPU agent worker 最多同时投递 16 个 teacher 请求，限制序列化暂存；完整 B2048 都会评分。
teacher 使用原生 COLOCATED level-1 sleep 保存冻结权重，student 使用原生 HYBRID 权重同步；
teacher 不加入 student 的参数同步列表。

四个 teacher 按固定 GPU 对逐个初始化并立即休眠，然后初始化 student rollout。
任一阶段失败终止当前 attempt，保留日志，从完整 checkpoint 恢复，不缩 batch、截断或丢样本。

## 身份、资格与 checkpoint

`prepare-execution` 在原 campaign 的 `executions/<identity>/` 创建不可变 runtime 和 record。
旧 manifest、launcher_repair、旧 controller 的状态及 checkpoint proof 不被改写。
新控制器的 STOP、guardian、states、CPU proof 和资格结果位于该 execution 目录。
正式训练仍使用原 `runs/<arm>`，保留模型、optimizer、LR、RNG、DataLoader、driver 及验证选择谱系。

两种节点配置的 student 仍是相同的 8 个逻辑 rank，无需把 student 的 checkpoint 从 16 rank 转成 8 rank。
新 runtime 的两种 profile 可以从同一完整状态继续。原版本已经明确不恢复 vLLM 采样 RNG 到逐位相同，
本版同样不承诺恢复后的新生成文本逐位一致。CPU 固定 token 测试检验评分、loss 和梯度契约；
实际 GPU 恢复需以下资格路线验证。

每个 record 必须明确选择一条路线，目标 profile 必须等于路线第二段：

| 路线 | 第一个新进程 | 第二个新进程 | 每臂更新数 | 每臂节点×更新数 |
|---|---|---|---:|---:|
| dual_node,single_node | 双节点 0→4，完整保存 | 单节点恢复 4→8 | 8 | 12 |
| single_node,dual_node | 单节点 0→4，完整保存 | 双节点恢复 4→8 | 8 | 12 |

两臂都跑一条路线共 16 次更新；两臂双向全部验证共 32 次更新、48 节点×更新数。
“节点×更新数”不是节点小时，时间需实测。默认不会追加另一条路线。
旧 B8/256-token 的短资格结果不能放行任何新 execution。
新资格保留完整训练数据与 200 步 sampler，只把本次运行终点设为 8、保存/验证间隔设为 4。
资格实际样本可能提前 EOS，因此通过后只能说明原预算下的实际批次通过，不能声称已经覆盖全批最坏 64K 长度。

## 本地检查与 ABCI 检查的边界

本地使用 pinned verl `482f86fd09959912c99c0acf2b90329f9e061da7`、Python 3.12、PyTorch 2.13 CPU，
执行真实 Hydra/dataclass、native teacher padding、固定 token loss/梯度、数据游标、保存恢复控制流与资格门禁测试。
GPU worker 的 CPU 测试使用文件替身；没有在本地声称 Megatron GPU 加载、H200 容量或吞吐已经通过。

macOS 上原有 Linux 专用测试有 9 项无法通过：6 个 ELF 链接测试、2 个 Linux GCC runtime replay、
1 个 Ray AF_UNIX 路径上限测试。旧 commit 在相同环境具有相同失败集合。
交付版整套结果为 388 PASS、3 FAIL、6 ERROR；旧版为 302 PASS、3 FAIL、6 ERROR，失败集合完全相同。
新增 86 项测试通过；最后针对性检查 95 项通过。可核对同目录 `mopd_node_profiles_cpu_validation.json`，
其中记录测试范围和被检查的 runtime 文件哈希。
ABCI 的 `preflight` 会实际检查原解释器、锁定包版本、GLIBC、GCC replay、tokenizer、数据与上游源文件哈希。
GPU 资格继续检查真实初始化、完整更新、保存、恢复及两种资源配置。

## 用户在 ABCI 拉取代码并准备执行快照

以下操作由用户在 ABCI 执行。本地交付没有部署或提交 GPU 作业。
新分支为 `codex/d4-mopd-node-profiles`。先在工作树干净时切到该分支：

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
git fetch --no-tags origin codex/d4-mopd-node-profiles
git switch codex/d4-mopd-node-profiles
git merge --ff-only origin/codex/d4-mopd-node-profiles
git log -1 --format='%H %s'
```

首次本地没有该分支时，`git switch` 会跟踪同名 origin 分支。若有未提交修改，先保存它们，不做 reset。

先准备“双→单”这一条路线。准备快照和 CPU preflight 可以在旧训练继续时执行：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_root="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3"
"$mopd_python" -u -I scripts/mopd.py prepare-execution \
  --manifest "$mopd_root/manifest.json" --profile single_node \
  --qualification-route dual_node,single_node > "$mopd_root/dual_to_single_execution.json"
mopd_record=$("$mopd_python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_path"])' \
  "$mopd_root/dual_to_single_execution.json")
"$mopd_python" -u -I scripts/mopd.py preflight \
  --manifest "$mopd_root/manifest.json" --execution-record "$mopd_record"
)
```

CLI 会自动进入 record 绑定的不可变 runtime；后续 checkout 变化不会替换已准备的代码。

## 在完整保存边界交接 GPU 资源

先查看两臂最新完整 checkpoint 与已经完成的 update，等到需要保留的更新已完成保存。
旧 `stop` 的语义是停止新提交并在下一次 update 前退出，恢复点仍是最近完整 checkpoint；
它不会立即为尚未保存的更新额外生成 checkpoint。

在选定保存边界请求停止旧控制器：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_manifest="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3/manifest.json"
env -u D4_MOPD_EXECUTION_RECORD "$mopd_python" -u -I scripts/mopd.py stop --manifest "$mopd_manifest"
env -u D4_MOPD_EXECUTION_RECORD "$mopd_python" -u -I scripts/mopd.py status --manifest "$mopd_manifest"
timeout 20s qstat -u "$USER"
)
```

等待旧 guardian 锁释放、旧 PBS allocation 终止。此命令不 qdel，不破坏正在进行的保存。
确认交接后提交资格测试，两臂各自自动完成第一段和第二段：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_root="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3"
mopd_record=$("$mopd_python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_path"])' \
  "$mopd_root/dual_to_single_execution.json")
"$mopd_python" -u -I scripts/mopd.py start --phase qualification --arm both \
  --manifest "$mopd_root/manifest.json" --execution-record "$mopd_record"
"$mopd_python" -u -I scripts/mopd.py status \
  --manifest "$mopd_root/manifest.json" --execution-record "$mopd_record"
)
```

所有新 PBS 脚本仍使用项目 `gcg51557`、reserved queue `R9920261000`、RTYPE `rt_HF`、`0390_mp_` 作业前缀。
路线第一段请求 `select=2`，第二段请求 `select=1`。
程序错误会 BLOCKED 等待分析；已有受证据约束的外部终止恢复规则继续使用。

反向验证是独立 record：准备时改为 `--profile dual_node --qualification-route single_node,dual_node`，
输出另存为 `single_to_dual_execution.json`；对该 record 重复 preflight 与 start qualification。
不要复用另一条路线的 receipt。先后顺序与资源占用由用户决定。

## 结果、显存与后续正式训练

每个 attempt 保存 `resolved.yaml`、`request.json`、`resume.json`、`result.json`、`update_*.json`、
`phase_events.jsonl` 和每节点 `gpu_memory_<rank>.csv`。
GPU 采样使用 UUID 标识物理卡，覆盖初始化、加载、保存和同步，间隔 250ms；报告的是采样峰值，
可能漏掉更短的瞬时峰值。资格完成前校验采样覆盖全部物理卡。各域 update 记录实际最大总 token 长度。

对状态输出中的实际 attempt 路径运行：

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python \
  -I scripts/mopd_profile_report.py /实际的/attempt/目录
```

比较时分开看初始化、验证、每步生成/评分/学习、保存与恢复；双节点的生成与评分本来重叠，
不能把两者相加当作双节点耗时。两种方式的每步时间与节点小时拿到后再决定正式采用哪种配置。

资格不会自动启动正式训练。确认结果和性能后，使用所选 record；以下以双→单的目标单节点为例：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d_mopd
mopd_python=/groups/gcg51557/experiments/0390_rlsd/envs/verl_mopd_482f86fd/.venv/bin/python
mopd_root="$PWD/outputs/mopd/qwen30b_d4_plain_r1_seed42_val1024_roll2_v3"
mopd_record=$("$mopd_python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_path"])' \
  "$mopd_root/dual_to_single_execution.json")
"$mopd_python" -u -I scripts/mopd.py start --phase train --arm both \
  --manifest "$mopd_root/manifest.json" --execution-record "$mopd_record"
)
```

正式提交会拒绝旧 guardian 仍持锁、其它 controller 仍可提交、含糊 pending submission、
未知/仍活动的旧 PBS allocation 或仍占用的 writer 锁。
原 campaign 的 STOP 可以保留，新执行使用自己的 STOP。再次切换 profile 前同样先停当前 record、
等待完整状态和旧 allocation 退出，再用已通过资格的目标 record 启动。
本版提供显式的 checkpoint 边界切换，不自动根据空闲节点抢占或迁移运行中的作业。
