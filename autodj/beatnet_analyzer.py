"""向后兼容模块：新版本离线分析已改用 Beat This!。"""

from .beat_this_analyzer import BeatThisAnalyzer

# 旧代码若仍导入 BeatNetAnalyzer，也会自动使用 Beat This!。
BeatNetAnalyzer = BeatThisAnalyzer

__all__ = ["BeatThisAnalyzer", "BeatNetAnalyzer"]
