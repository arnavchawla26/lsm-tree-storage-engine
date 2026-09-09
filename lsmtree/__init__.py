"""lsmtree: a from-scratch LSM-tree key-value storage engine.

Public API:
    from lsmtree import LSMTree
"""

from .engine import LSMTree, NOT_FOUND

__all__ = ["LSMTree", "NOT_FOUND"]
__version__ = "0.1.0"
