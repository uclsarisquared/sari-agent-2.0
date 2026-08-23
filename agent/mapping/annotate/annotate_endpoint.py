"""Provider-neutral endpoint annotator.

The implementation remains importable as ``annotate_qwen`` for compatibility with old scripts.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAPPING_DIR = os.path.dirname(_THIS_DIR)
for _path in (_MAPPING_DIR, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import _bootstrap  # noqa: F401,E402

from annotate_qwen import (  # noqa: F401
    DEFAULT_MODEL,
    EndpointAnnotateError,
    annotate,
    main,
)

if __name__ == "__main__":
    main()
