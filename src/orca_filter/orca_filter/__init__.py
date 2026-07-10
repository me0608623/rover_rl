"""orca_filter — 純 ORCA 安全過濾器(車端部署版)。

從 IsaacLab play_eval/rvo2_safety_filter.py 抽取,無 Isaac 依賴。
non-cooperative 設計(行人不會避讓機器人):ego_responsibility=1.0 + obs_inflation=1.8。
"""
from .orca_core import ORCAFilter, ORCAInput, ORCAOutput

__all__ = ["ORCAFilter", "ORCAInput", "ORCAOutput"]
