#!/usr/bin/env python3
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patched auditwheel script.

Patch changes behavior of auditwheel to not remove libpython from the wheel
as it is python interpreter library required by python backend.
"""

import re

import auditwheel.main  # noqa
import auditwheel.policy.external_references
from auditwheel.policy import _POLICIES as POLICIES

# to not remove libpython from the wheel as it is python interpreter library required by python backend
# used here: https://github.com/pypa/auditwheel/blob/main/src/auditwheel/policy/external_references.py#L28
auditwheel.policy.external_references.LIBPYTHON_RE = re.compile(
    r"__libpython\d\.\d\.\d\.so"
)

# Policies to ignore attaching Python libraries to wheel during fixing dependencies
for p in POLICIES:
    for version in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
        p["lib_whitelist"].append(f"libpython{version}.so.1.0")

if __name__ == "__main__":
    auditwheel.main.main()
