# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Insert / maintain / migrate to an SPDX license header on source files.

Pre-commit hook that adopts the two-line SPDX header on every file it is run
against. Because pre-commit runs hooks on the files staged in a commit, scoping
this hook by file type (rather than by directory) migrates each source file to
SPDX *the first time it is touched* -- a low-risk, incremental rollout.

The header uses a copyright *year range*, mirroring the repo convention (the
``LICENSE`` file and ``add_copyright.py`` use ``<start>-<current>``)::

    # SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: BSD-3-Clause

Per file, the hook does exactly one of:

* **Maintain** -- already SPDX: bump the copyright end year to the current year.
* **Migrate** -- carries the legacy long-form NVIDIA BSD header: replace that
  whole block in place with the two SPDX lines, preserving the comment style
  (``#`` or ``//``), the shebang, and the block's start year.
* **Insert** -- no NVIDIA header: add the SPDX header (after a shebang if any).

Idempotent, and coexists with ``add_copyright.py`` (which runs first and only
maintains the copyright year on the legacy/SPDX string it recognizes).
"""

import os
import re
import sys
from datetime import datetime

ROOT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
LICENSE_PATH = os.path.join(ROOT_DIR, "LICENSE")
CURRENT_YEAR = str(datetime.now().year)

_SPDX_MARKER = "SPDX-License-Identifier"
_LICENSE_ID = "BSD-3-Clause"

# Start year of the project-wide LICENSE (used only when inserting into a file
# that has no NVIDIA copyright at all).
_LICENSE_YEAR_PAT = re.compile(r"Copyright \(c\) (\d{4})(?:-\d{4})?, NVIDIA")
# Existing SPDX copyright line (for year maintenance).
_SPDX_YEAR_PAT = re.compile(
    r"(SPDX-FileCopyrightText: Copyright \(c\) )(\d{4})(?:-\d{4})?(, NVIDIA)"
)
# First line of a legacy long-form NVIDIA BSD header (captures comment prefix +
# start year). The block ends at the standard BSD "SUCH DAMAGE" line.
_LEGACY_COPY_PAT = re.compile(
    r"^(#|//) ?Copyright(?: \(c\))? (\d{4})(?:-\d{4})?,? NVIDIA CORPORATION"
)
_LEGACY_END_MARK = "POSSIBILITY OF SUCH DAMAGE"

_CPP_EXTS = (".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh")


def _year_range(start):
    return start if start == CURRENT_YEAR else "{}-{}".format(start, CURRENT_YEAR)


def _license_start_year():
    try:
        with open(LICENSE_PATH, "r", encoding="utf-8") as f:
            match = _LICENSE_YEAR_PAT.search(f.read())
        if match:
            return match.group(1)
    except OSError:
        pass
    return CURRENT_YEAR


def _spdx_lines(prefix, years):
    return (
        "{p} SPDX-FileCopyrightText: Copyright (c) {y}, NVIDIA CORPORATION "
        "& AFFILIATES. All rights reserved.\n"
        "{p} SPDX-License-Identifier: {lic}\n".format(
            p=prefix, y=years, lic=_LICENSE_ID
        )
    )


def _maintain(content):
    def _bump(match):
        return match.group(1) + _year_range(match.group(2)) + match.group(3)

    return _SPDX_YEAR_PAT.sub(_bump, content)


def _migrate_legacy(content):
    """Replace a legacy long-form NVIDIA BSD header with the SPDX lines. Returns
    the new content, or None if no legacy header is found."""
    lines = content.splitlines(keepends=True)
    start = prefix = start_year = None
    for i, line in enumerate(lines):
        match = _LEGACY_COPY_PAT.match(line)
        if match:
            start, prefix, start_year = i, match.group(1), match.group(2)
            break
    if start is None:
        return None
    end = None
    for j in range(start, len(lines)):
        if _LEGACY_END_MARK in lines[j]:
            end = j
            break
    if end is None:
        return None
    header = _spdx_lines(prefix, _year_range(start_year))
    return "".join(lines[:start]) + header + "".join(lines[end + 1 :])


def _insert(path, content):
    prefix = "//" if path.endswith(_CPP_EXTS) else "#"
    header = _spdx_lines(prefix, _year_range(_license_start_year())) + "\n"
    lines = content.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        return lines[0] + header + "".join(lines[1:])
    return header + content


def process(path):
    """Bring ``path`` to an SPDX header (maintain/migrate/insert). Returns True if
    the file was modified."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return False

    if _SPDX_MARKER in content:
        updated = _maintain(content)
    else:
        updated = _migrate_legacy(content)
        if updated is None:
            updated = _insert(path, content)

    if updated == content:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return True


def main(argv):
    changed = False
    for path in argv[1:]:
        if process(path):
            print("Updated SPDX header: {}".format(path))
            changed = True
    # Non-zero exit tells pre-commit the file was modified so it re-stages.
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
