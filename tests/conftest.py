"""pytest 共享配置：保证未安装时也能从项目根导入 kairos_crypto（全程离线）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
