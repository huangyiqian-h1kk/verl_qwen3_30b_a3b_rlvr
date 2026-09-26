# 多领域实验依赖及验收范围

从项目根目录，激活原 conda 后运行：

```bash
bash scripts/multidomain/setup_cpu_dependencies.sh
source scripts/multidomain/env.sh
python -m unittest discover -s tests/multidomain -v
```

安装器复用 `.venv-multidomain/`，先采集原 conda 的包版本、位置和 `pip check` 结果，再生成保护训练依赖的 constraints。评分包及其专用依赖可在项目环境中另装；bootstrap 工具 pip/setuptools/wheel 不纳入训练版本约束。安装后核对原环境未出现包版本或位置变化，且项目没有替换、遮蔽或丢失受保护的原环境包。

## 评分版本来源

采用 NeMo Gym commit `6283f37f83f727561ce59e0012c00dcd9780a423`：

| 包 | 版本 | 依据 |
|---|---|---|
| math-verify | 0.8.0 | `resources_servers/math_with_judge/requirements.txt` |
| latex2sympy2_extended | 1.10.2 | math-verify 0.8.0 的发布包依赖 |
| openapi-schema-validator | 0.6.3 | `resources_servers/structured_outputs/requirements.txt` |
| xmltodict | 1.0.2 | 同上；此前 `<1` 的范围错误，已纠正 |
| reasoning-gym | 0.1.25 | 本轮任务/参考答案验证使用的固定版本；上游要求为 `>=0.1.19` |

其余 requirements 中的版本范围是项目兼容范围声明，未穷举验证其中全部版本。实际训练依赖由原 ABCI 环境的版本约束决定。这里只接入选定的评分函数，没有重建整个 NeMo Gym HTTP 服务环境。

## 已确认的原环境问题

2026-09-26 用户在 ABCI 分别读取原 conda 与项目环境，确认：

| 依赖声明问题 | 处置 |
|---|---|
| outlines 0.1.11 要求 outlines_core 0.1.26，实际为 0.2.11 | 保留原包；vLLM 0.11.0 明确要求 outlines_core 0.2.11，不能直接降级 |
| megatron-core 0.13.1 要求 NumPy <2，实际为 2.2.6 | 保留原训练组合；旧项目有 `np.product` 兼容处理，但仍需 GPU 运行验证 |
| 原环境报告 decord 0.6.0 不支持当前平台 | 持续记录；项目 pip 未报告同一提示不等于问题已修复 |

验收器只接受以上**准确诊断与准确包版本**，且必须在原环境本次快照中存在；不接受任意旧环境错误。未知问题、新增问题、受保护包被替换或遮蔽、原环境发生变化、评分版本错误，均返回 FAIL。它没有修改包的依赖声明，也没有使用 `pip check || true` 宣告成功。

有这些基线问题时，成功状态为 `CPU_DEPENDENCIES_READY_WITH_INHERITED_CONFLICTS`。这只允许继续 CPU 数据准备和评分测试，不能用作 GPU/模型/训练验收结论；GPU 阶段仍按项目既定基础设施、模型 smoke、backward、baseline 的验收顺序进行。

## 输出

- `.venv-multidomain/base.before.json`：安装前原环境快照。
- `.venv-multidomain/base.constraints.txt`：训练依赖版本约束。
- `.venv-multidomain/dependency_audit.json`：安装前后原环境、项目环境、未解决问题及验收结果。
- `.venv-multidomain/resolved.requirements.txt`：成功安装后的实际包版本清单；它不能替代 editable 源码 commit、ABCI modules 和 GPU 运行记录。

当前 CPU 单测共 33 项，其中 9 项验证依赖验收规则。ABCI 结果以实际执行输出为准。
