# Test report — Auto DJ 1.2.18 research-guided optimization

## Automated tests

```text
94 passed in 12.49s
```

新增专项覆盖：

1. 完全相关素材的 equal-power 中点增益由约 1.414 修正到约 1.0，首尾端点不变。
2. 能量上下受控反转优于连续大幅同向移动。
3. 小队列精确子集 DP 避免高首边诱导的贪心死端。
4. 末尾安全路径选择能够独立承接的 IN cue，而非安静尾部的高置信度 cue。

## Performance smoke benchmark

测试环境中的合成队列排序：

```text
8 tracks   0.015 s
9 tracks   0.033 s
10 tracks  0.129 s
11 tracks  0.040 s
20 tracks  0.404 s
```

10 首及以下使用精确 DP；11 首起使用有界 beam search。所有结果包含每首曲目且首曲固定。

## Independent continuity checks

- `check_natural_transition.py`: PASS
- `check_mixend_continuity.py`: PASS
- `check_drum_loop_bridge.py`: PASS
