"""
Code execution engine for GDPval.

Provides a persistent Python namespace (like Jupyter) where variables persist
across calls within an episode. Uses ``exec()`` with stdout/stderr capture
and thread-based timeout.

WARNING: Uses in-process exec() with full builtins. Model-generated code has
unrestricted host access. Use only in trusted/sandboxed environments.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Process-wide lock for os.chdir() — CWD is process-global state
_CWD_LOCK = threading.Lock()

# Common imports to pre-seed in every namespace
_PRESEEDED_CODE = """\
import os
import json
import math
import re
import csv
import datetime
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass
"""


class CodeExecutor:
    """Persistent Python namespace executor for GDPval episodes.

    Each episode gets a fresh namespace. Variables persist across ``execute()``
    calls within the same episode (like Jupyter cells).

    Args:
        workdir: Working directory for the episode. All code runs with cwd set here.
        timeout_sec: Maximum time for each code execution.
    """

    def __init__(self, workdir: Path, timeout_sec: int = 60) -> None:
        self.workdir = workdir
        self.timeout_sec = timeout_sec
        self._namespace: dict[str, Any] = {}
        self._leaked_threads = 0
        self._initialize_namespace()
        logger.warning(
            "gdpval: CodeExecutor uses in-process exec() with full builtins. "
            "Model-generated code has unrestricted host access. "
            "Use only in trusted/sandboxed environments."
        )

    def _initialize_namespace(self) -> None:
        """Pre-seed the namespace with common imports."""
        self._namespace = {"__builtins__": __builtins__}
        # Run pre-seeded imports silently
        try:
            exec(_PRESEEDED_CODE, self._namespace)  # noqa: S102
        except Exception as e:
            logger.warning("gdpval: failed to pre-seed namespace: %s", e)

    def execute(self, code: str) -> tuple[str, str, int]:
        """Execute Python code in the persistent namespace.

        Args:
            code: Python code to execute.

        Returns:
            ``(stdout, stderr, return_code)`` where return_code is 0 on success.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        return_code = 0
        exception_info: list[str] = []

        # Save CWD before thread starts so we can always restore it
        with _CWD_LOCK:
            saved_cwd = os.getcwd()

        def _run():
            nonlocal return_code
            with _CWD_LOCK:
                os.chdir(self.workdir)
            try:
                with (
                    contextlib.redirect_stdout(stdout_buf),
                    contextlib.redirect_stderr(stderr_buf),
                ):
                    exec(code, self._namespace)  # noqa: S102
            except Exception as e:
                return_code = 1
                exception_info.append(f"{type(e).__name__}: {e}")
            finally:
                with _CWD_LOCK:
                    try:
                        os.chdir(saved_cwd)
                    except (FileNotFoundError, OSError):
                        pass

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive():
            self._leaked_threads += 1
            if self._leaked_threads >= 5:
                logger.error(
                    "gdpval: %d leaked threads from timeouts — "
                    "model may be generating infinite loops",
                    self._leaked_threads,
                )
            # Restore CWD from main thread since the timed-out thread may
            # have left us in the workdir (which may get cleaned up)
            with _CWD_LOCK:
                try:
                    os.chdir(saved_cwd)
                except (FileNotFoundError, OSError):
                    pass
            return_code = 124
            stderr_text = stderr_buf.getvalue()
            stderr_text += f"\n[Execution timed out after {self.timeout_sec}s]"
            return stdout_buf.getvalue(), stderr_text.strip(), return_code

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        if exception_info:
            stderr_text = (stderr_text + "\n" + "\n".join(exception_info)).strip()

        return stdout_text, stderr_text, return_code

    def reset(self) -> None:
        """Clear the namespace for a new episode."""
        self._namespace.clear()
        self._initialize_namespace()
