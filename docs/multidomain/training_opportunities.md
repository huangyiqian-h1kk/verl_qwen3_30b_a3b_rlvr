# 单节点运行期间择机转双节点

入口仍为 `scripts/multidomain/manage_training_continuation.py`。
`enable-opportunistic --root ROOT` 升级两个已接管的 continuation 监护；不重新
执行最初的 `apply`，不追加监护时间，不修改已提交的训练作业快照。

## 启用后的行为

| 条件 | 行为 |
| --- | --- |
| 正式作业为 R、实际一节点，已有完整 8-rank checkpoint，至少还剩五步 | 每 300 秒观察一次资源 |
| 连续两次观察到至少两个额外可用整节点 | 创建独立候选，提交到 `R9920261000`，24 小时 walltime，无用户 hold |
| 正式作业为 Q/H、已在双节点、已结束训练、恢复被应用错误阻塞 | 不创建资源机会候选 |
| 容量未知、队列余量不足或节点不足 | 重置连续计数，原训练和通常的中断重投继续 |
| domain 和 random 同时等待资源 | domain 优先；已提交但尚未运行的候选预占两个名额；运行中的候选不重复扣除 |
| 候选被管理员终止（271 且 PBS 确认 root 终止），或明确超过 walltime（-29） | 保留／恢复本 setup 单节点；冷却 30 分钟，再等待连续两次容量确认 |
| 候选已有 FAIL 结果、应用异常或验证不一致 | 本 setup 停止自动尝试升级；继续单节点监护，另一 setup 独立进行 |
| 候选状态未知或 qsub 返回不明确 | 保留原单节点；核实同一提交记录，不盲目再 qsub |
| 双节点成功接手 | 保持运行，不因空闲资源减少而主动降回单节点 |
| 成功交接后的正式训练被 271 / 明确 walltime -29 终止 | 保留原重投规则：至少一个可用整节点则投双节点，否则／容量未知投单节点 |
| 重投后单节点已运行并保存完整 8-rank checkpoint | 可以再进入本机制，不必等下一次被 kill |

“两个空闲节点”是当前正式单节点以外、可供候选同时使用的资源。计数沿用已核验
的 reserved queue/project/`nii` 规则：合并 NUMA vnodes 为 192 CPU/8 GPU 的完整物理
节点，同时受队列 CPU 余量约束。共享准入锁将另一 setup 的待提交／排队候选算入
需求。它不是整个集群的空闲节点数，也不是保证资源已被预留；PBS 仍决定最终分配。

同一 setup 同时最多一个新机会候选。另一个 setup 的正常训练不等待它验证成功。
domain 优先只适用于仍有活动监护、心跳新鲜、单节点正常运行且当前可尝试升级的
domain；应用失败、冷却中或没有合适 checkpoint 的 domain 不长期占住优先权。

旧版本留下的 `SINGLE_ONLY` 可在此升级后重新尝试，但只接受能够重新核验的
管理员 271／walltime 中断，且该作业没有 FAIL 结果。用户手动取消、验证失败、
历史 PBS 证据不可用都不会被猜测成“只是没来得及”。

## 交接仍有完整验证

每个新候选使用新的 attempt ID 和 checkpoint lineage。在它真正开始运行时，从
当时正式训练的最新完整 checkpoint 获取源状态。沿用同一个双节点 PBS allocation
内的五个新进程验证阶段：A 8→16、B 16→16、C 16→8、D 8→8、E 8→16；各阶段
实际完成原计划下一批训练、完整保存和 rollout 权重同步。domain 仍检查采样计划、
prompt ID、dataloader 游标连续性。全局 batch、reward 和数据快照不改变。

只有完整验证通过且候选确为当前运行的 PBS execution，才保护单节点 checkpoint、
停止本 setup 当前单节点、等待其退出并取得 writer lease、保护最终最高完整
checkpoint。候选必须追到明确的目标 step，监护写入持久交接 commit 后才成为正式
训练来源，并在同一作业内继续运行。失败候选的 checkpoint 不会自动变为正式来源。

若停止单节点后候选失败，先确认双方终止，再从最高完整单节点 checkpoint 恢复
一节点训练，即使这时监护已到截止时间，也完成已经需要的回退。若 commit 已落盘，
重启监护会恢复这个交接决定，不把旧单节点重新当作权威来源。

本次保留完整五阶段验证，不增加“只凭以前通过记录就跳过恢复验证”的快捷路径。
因此每次尝试有验证和追赶成本。仍按完整 checkpoint 交接，不热修改当前 trainer，
不承诺保留尚未保存的步骤；这些步骤可能需要重算。

## 升级与核验

从交接回复中的固定 Git commit 更新已有运维 worktree。在当前监护所在登录节点
运行（本次上传状态为 `login3`）；程序会核验本机、用户、精确进程入口和所有权锁。

```bash
"$D4_PY" -I "$D4_SCRIPT" self-test
"$D4_PY" -I "$D4_SCRIPT" enable-opportunistic --root "$D4_ROOT"
"$D4_PY" -I "$D4_SCRIPT" status --root "$D4_ROOT"
qstat
```

升级先核验并冻结两套新 runtime，保留原 reward、audit、配置和其他快照；再分别
温和停止登录节点监护，持有其锁时原子更新计划，启动新监护。使用现有 pidfd
兼容实现，不使用数字 PID 的盲目 kill，不对训练作业发升级用 qdel。
deadline 不变，正在运行的双节点作业继续使用原快照。

成功回执是 `OPPORTUNISTIC_MONITORS_RUNNING`，两个 setup 的
`controller_runtime.verified` 均为 true，并绑定实际导入的 opportunities/controller
代码。若 15 秒内未确认新监护完成首轮检查，返回
`OPPORTUNISTIC_MONITORS_UNVERIFIED` 和非零退出码；查看 receipt/status 中的原因，
可重复同一启用命令。已经正确运行的监护不重复重启，也不再延长时间。

`status` 新增 `opportunistic_policy`、`opportunity_poll` 和 `opportunity`：

| 字段／值 | 含义 |
| --- | --- |
| `opportunity_poll.reason=waiting_second_capacity_observation` | 第一次容量满足，等待下一轮确认 |
| `fewer_than_two_free_slots` / `capacity_or_eligibility_unknown` | 资源不足／证据不明，单节点继续 |
| `domain_first` | 等待可升级的 domain 优先取得准入 |
| `opportunity.phase=SUBMITTING` | 准入已落盘，提交可能尚无 PBS receipt |
| `QUALIFYING` | 已有候选，等待运行及完整验证；查询 PBS 可区分 Q/R |
| `STOPPING_SINGLE` / `CATCH_UP` | 已验证，正在安全交接 |
| `COMMITTED` | 本次交接已持久确认 |
| `FAILED` + `failure_class=external` | 外部中断；`retry_after_utc` 后可重新观察容量 |
| `FAILED` + 其他 failure class | 自动升级已阻塞；保留具体错误和单节点恢复 |

每次尝试的 journal、回退保护和 commit 位于
`control/opportunities/<attempt-id>/`；旧 `control/handoff_commit.json` 保留历史意义。
判断后续交接以当前 `state.authority` 和当前 attempt 的 commit 为准。

## 验证范围

CPU 与进程级测试覆盖资源去抖、domain 优先与排队需求、重复提交防护、271 冷却、
验证错误、两套独立状态、交接中断回退、无 exit code 的已确认取消、checkpoint
保护中断、commit 后重启、双→单→再双以及真实后台进程的快照升级和导入绑定。
测试使用合成 PBS／训练数据；实际 GPU 恢复仍须由候选五阶段验证完成后才允许交接。
