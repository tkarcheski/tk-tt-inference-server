# SPDX-License-Identifier: Apache-2.0
"""Record-aware log chunking.

Splits on lines that start a new log record (timestamped), so stack traces
and multi-line records never straddle a chunk boundary. Chunks are sized for
a small model: the window must stay well inside what a 0.5B attends to well.
"""

import re
from pathlib import Path
from typing import Iterator

# Lines that begin a new log record in this repo's logs:
#   "2026-02-12 23:06:33,940 - run.py:468 - INFO: ..."  (workflow logs)
#   "INFO 02-13 05:11:51 [__init__.py:250] ..."          (vLLM logs)
RECORD_START = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    r"|(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) \d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)

MAX_LINES = 80
MAX_CHARS = 6000  # ~1.5k tokens; keeps prompt small for a 0.5B model


def iter_chunks(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (first_line_number, chunk_text) for one log file."""
    lines: list[str] = []
    start_line = 1
    with open(path, errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            boundary = RECORD_START.match(line) and (
                len(lines) >= MAX_LINES or sum(map(len, lines)) >= MAX_CHARS
            )
            # Hard cap for files with no recognizable record starts.
            if boundary or len(lines) >= 4 * MAX_LINES:
                yield start_line, "".join(lines)
                lines, start_line = [], lineno
            lines.append(line)
    if lines:
        yield start_line, "".join(lines)
