# Research Notes

## Beat/downbeat front end

0.5 版本使用 Beat This! 1.1.0 作为离线主分析器：

1. File2Beats 输出 beat 和 downbeat。
2. infer_beat_numbers 生成拍号编号。
3. 对稳定电子音乐网格做保守的重复峰合并与漏拍补全。
4. Beat This! 不使用 DBN，因此不会对速度变化歌曲强制固定 tempo/meter。
5. 所有 OUT/IN 和 phrase 搜索均基于 Beat This! downbeat。

## Continuous tempo recovery

使用 source sample 到 target sample 的单调 key-frame map：

- 恢复前斜率为 `1 / initial_rate`
- 恢复区局部 rate 使用五次 smootherstep 和几何插值
- 恢复后斜率为 1
- Rubber Band R3 使用 time-map 一次性渲染
- 回退算法在单个 STFT 中按 target->source 时间坐标采样

这避免了逐小节独立 phase vocoder 的重启和短 crossfade 接缝。
