import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bifusion3d  # noqa: E402,F401  (anade modelos/mamba a sys.path)
