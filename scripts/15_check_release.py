"""新克隆也可在安装科学计算依赖前运行的只读发布检查。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erp_core_p3b.release_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
