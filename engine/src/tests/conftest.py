"""Test configuration for the engine service.

Adds the src directory to sys.path so that absolute imports (from app...) work
when pytest is run from engine/src/ or the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure engine/src is on the path so `from app...` imports resolve.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
