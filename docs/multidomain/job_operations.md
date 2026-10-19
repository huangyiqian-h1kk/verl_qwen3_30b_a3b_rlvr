# 四域训练：统一诊断与 Git 更新

代码仓库：`huangyiqian-h1kk/verl_qwen3_30b_a3b_rlvr`。
四域分支：`research/data-mixture-rl-4d`。默认 `main` 是另外的实验入口，不用于部署四域修复。

## 查看当前作业、历史退出与交接状态

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d
./.venv-multidomain/bin/python -I scripts/multidomain/diagnose_jobs.py --root "$PWD"
```

重复 `--job JOB_ID` 可补充已经退出的历史作业；默认也读取两边监护引用的作业、
最近交接事件涉及的作业、两个单节点/两个双节点/公共 check 各自最新的作业记录。
`--tail-lines 80` 可以增大日志尾部。所有结果输出到终端；不创建报告、补丁或压缩包。

新工具尚未合并/安装时，可以从修复分支取出一个脚本运行，无需切换集群工作树：

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/data_mixture_rl_4d
git fetch --no-tags https://github.com/huangyiqian-h1kk/verl_qwen3_30b_a3b_rlvr.git \
  refs/heads/codex/d4-job-diagnostics
git show FETCH_HEAD:scripts/multidomain/diagnose_jobs.py |
  ./.venv-multidomain/bin/python -I - --root "$PWD"
)
```

`git fetch` 只更新 Git 对象和 FETCH_HEAD；`git show` 直接提供脚本内容。
不要为运行这个诊断 checkout、reset、pull 或覆盖正在使用的训练模块。
核对提交时，可把 `FETCH_HEAD` 替换成已审阅的完整 commit SHA。

诊断脚本只用 Python 标准库，不导入项目训练、Ray、torch、判分器或监护入口。
只执行 `qstat` 和 Git 读取命令；不调用 `qsub`、`qdel`、监护 resume 或更改配置。
锁探针仅打开已有锁文件，不创建锁或状态目录。Git status 关闭可选的索引写锁。
日志扫描限制在每个文件最后 256 KiB，JSON 上限 2 MiB，不遍历/加载 checkpoint tensors。
三个并发只读 PBS 查询分别有 15 秒超时，查询失败单独显示，后续证据继续收集。

## 解释结果

- `jobs.*.pbs`：PBS 历史状态、退出码、comment、申请/使用 walltime。查不到历史
  只能记作未知，不推断训练成功，不据此重复提交。
- `controllers.*.state`：交接阶段、截止时间、候选和单节点 lineage、阻塞原因。
- `controllers.*.lock.held`：该瞬间能否观察到控制器排他锁；记录中的 RUNNING
  可能过时。不能在 login4 用 login1 的 PID 直接判断进程存活。
- `records`：公共 PASS、单节点取消日志、放行决定和最终交接记录。诊断只展示这些
  记录，不重新验证证据哈希，也不授权切换或删除 checkpoint。
- `recent_events` 与 `log`：区分正常交接、271 重试、应用失败和监护退出。
- `single_monitors`：单节点旧监护状态，包含是否被交接控制器接管和待确认提交。
- `jobs.*.records.*.reports`：候选 8→16、16→16 各阶段已有结果；排队时没有
  result.json 是正常情况，缺文件本身不能证明失败。

这是非原子诊断快照，收集期间调度状态可能变化。UTC 采集时间明确标出，PBS 自带的
时间字符串原样保留。只有新 domain 单节点出现在 qstat 中时，还不能确认它由哪个
监护重投，或另外三个作业为什么消失；需要结合上述证据。

## 后续修复和目录整理

1. 每次修复在四域分支上建立 Git 分支、提交和 PR；变更和验证结果集中记录在仓库。
   日志、数据、模型、checkpoint、环境和集群上的绝对路径记录不作为源码提交。
2. 当前集群使用的日期包与 Git 四域分支存在差异。先读取实际部署的监护 entry、plan、
   runtime、单节点提交入口和包哈希，再把确认过的实现合并到固定源码目录。
   本次 PR 只新增诊断工具及文档，不声称已经迁移/修复当前训练运行时。
3. 活跃候选的代码快照、identity、PASS 绑定和 checkpoint 引用必须保持一致。
   新 commit 不等于当前进程已经更新；部署需要明确记录版本并协作切换监护。
4. 后续运维统一放在 `scripts/multidomain/`，说明放在 `docs/multidomain/`。
   输出保留在 `outputs/`。停止向项目根目录追加日期命名补丁包。
5. 清理历史包前，检查现有 PBS 脚本、所有可恢复监护的 plan/entry/runtime、
   回退 checkpoint 和外部代码路径的引用。未完成引用核对前不移动或删除历史包；
   归档也会改变路径，不能直接批量 mv。`git clean -fd`/`-fdx` 不用于这个工作目录。

## 本地验证

```bash
python3 -m unittest discover -s tests/multidomain -p test_job_diagnostics.py -v
```

这些测试使用模拟 PBS 结果与临时运行目录，不调用真实调度器，不提交 GPU 作业。
