# Logic30 的 DeepMath test 补测

Logic30 的原四域 test 只有 Nemotron Math；新增的独立补测复用已经冻结的
DeepMath `test.parquet`（本 campaign 为 256 题，L8/L9 为 213/43），
与 base、Math50、IF70、CP80 的 DeepMath test 保持同一数据和 verifier。
模型必须来自 `runs/logic_algorithmic/export.json` 中经过加载验证的 step 30。

先在仓库根目录设置 `teacher_python` 为现有 campaign 使用的 Python，
`teacher_campaign` 为现有 campaign 目录。无需切换到原 guardian 所在节点。

```bash
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py start --campaign "$teacher_campaign" --logic30-test
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py status --campaign "$teacher_campaign" --logic30-test
# 完成后核验逐题 journal 并汇总整体、L8、L9 分数：
"$teacher_python" -I scripts/evaluate_teacher_deepmath.py report --campaign "$teacher_campaign" --logic30-test
```

任务名 `deepmath_logic30`，PBS 名 `0390_dm_la30`，1 节点/8 GPU、单次
walltime 24 小时。计划单独发布为 `deepmath_logic30_test.json`，完成收据为
`runs/deepmath_logic30/heldout.json`，汇总为
`deepmath_evaluation/logic30_comparison.json`。原训练、选择记录、HF 导出、四域
test 和既有 DeepMath 计划均保留；不重新训练、导出或抽样，也不运行 DeepMath
validation。总览脚本会列出新任务及结果。

`start` 从任意登录节点执行，沿用共享锁交接和
`domain-teacher-external-interruption-v2` 监护。已全部完成、guardian 已退出的
campaign 会重新启动监护；重复执行会接管同一任务，不重复投递。确认 271、
walltime 超限或符合现有严格证据检查的 root 265 终止后，自动续投并复用匹配
身份的已完成批次。PBS 查询未知则保留原 job 持续观察；应用异常保持阻塞。

CPU 测试覆盖精确模型/数据绑定、原计划保留、断批恢复、逐题收据、271/265
恢复、未知 PBS 防重复、CLI/总览，以及真实 guardian 子进程在已完成 campaign
上的重启和重复启动。GPU、权重与 PBS 使用 fixture；集群提交和完成以实际
job ID、完整收据为准。
