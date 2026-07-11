# 1.2.18 升级说明：相关性感知淡化与上下文序列搜索

## 兼容性

本版本不改变缓存格式、模型名称、设置文件格式或音频文件要求。可以直接覆盖 1.2.17 代码运行，旧分析缓存仍可使用。

## 转场算法

简化转场与尾端安全转场现在会在规划阶段测量低频、中频和高频重叠区的实际相关性。对于正相关信号，算法根据

```text
P = a² + b² + 2ρab
```

中的交叉项修正两条增益曲线，使原始 fade law 的目标功率保持不变。曲线端点不移动，最大衰减约 3 dB，负相关不自动提升。

尾端路径不再要求下一首 cue 前有常规鼓循环 pre-roll。所有入口仍必须来自 CUE-DETR，但会额外评估 cue 后空间、局部能量、onset 和 vocal 风险，避免进入下一首安静尾部。

## 排序算法

- `N <= 10`：精确子集动态规划，保留累计目标和最弱边的 Pareto 状态。
- `N > 10`：两步前瞻 beam search，并给不同末端节点保留少量搜索槽位。
- 完整路径后继续执行确定性 swap/relocate 改良。
- 能量轨迹从“惩罚升降交替”修正为“奖励受控反转、惩罚连续大幅同向变化”。

## API

公开调用接口没有变化。`TransitionPlan.metrics` 新增：

- `fade_correlation_low`
- `fade_correlation_mid`
- `fade_correlation_high`
- `correlation_compensation_db`
- 尾端路径下的 `tail_entry_quality`
