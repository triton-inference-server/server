#!/usr/bin/env python3

# Copyright 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import pathlib
import re

FLAGS = None
SKIP_EXTS = (
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".log",
    ".preprocessed",
    ".jmx",
    ".gz",
    ".json",
    ".pdf",
    ".so",
    ".onnx",
    ".svg",
    "pull_request_template.md",
)
REPO_PATH_FROM_THIS_FILE = "../.."
SKIP_PATHS = (
    "build",
    "deploy/gke-marketplace-app/.gitignore",
    "deploy/gke-marketplace-app/server-deployer/chart/.helmignore",
    "deploy/gcp/.helmignore",
    "deploy/aws/.helmignore",
    "deploy/fleetcommand/.helmignore",
    "docs/.gitignore",
    "docs/_static/.gitattributes",
    "docs/examples/model_repository",
    "docs/examples/jetson",
    "docs/repositories.txt",
    "docs/exclusions.txt",
    "docker",
    "qa/common/cuda_op_kernel.cu.cc.patch",
    "qa/ensemble_models/mix_platform_float32_float32_float32/output0_labels.txt",
    "qa/ensemble_models/mix_type_int32_float32_float32/output0_labels.txt",
    "qa/ensemble_models/mix_ensemble_int32_float32_float32/output0_labels.txt",
    "qa/ensemble_models/wrong_label_int32_float32_float32/output0_labels.txt",
    "qa/ensemble_models/label_override_int32_float32_float32/output0_labels.txt",
    "qa/L0_model_config/noautofill_platform",
    "qa/L0_model_config/autofill_noplatform",
    "qa/L0_model_config/autofill_noplatform_success",
    "qa/L0_model_config/special_cases",
    "qa/L0_model_config/cli_messages/cli_override/expected",
    "qa/L0_model_config/cli_messages/cli_deprecation/expected",
    "qa/L0_model_config/model_metrics",
    "qa/L0_model_namespacing/test_duplication",
    "qa/L0_model_namespacing/test_dynamic_resolution",
    "qa/L0_model_namespacing/test_ensemble_duplication",
    "qa/L0_model_namespacing/test_no_duplication",
    "qa/L0_perf_nomodel/baseline",
    "qa/L0_perf_nomodel/legacy_baseline",
    "qa/L0_warmup/raw_mug_data",
    "qa/L0_java_resnet/expected_output_data",
    "qa/L0_trt_dla_jetson/trt_dla_model_store",
    "qa/openvino_models/dynamic_batch",
    "qa/openvino_models/fixed_batch",
    "CITATION.cff",
    "TRITON_VERSION",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE",
)

COPYRIGHT_YEAR_RE = "Copyright( \\(c\\))? 20[1-9][0-9](-(20)?[1-9][0-9])?(,((20[2-9][0-9](-(20)?[2-9][0-9])?)|([2-9][0-9](-[2-9][0-9])?)))*,? NVIDIA CORPORATION( & AFFILIATES)?. All rights reserved."

COPYRIGHT = """

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

repo_abs_path = (
    pathlib.Path(__file__).parent.joinpath(REPO_PATH_FROM_THIS_FILE).resolve()
)

copyright_year_re = re.compile(COPYRIGHT_YEAR_RE)


def visit(path):
    if FLAGS.verbose:
        print("visiting " + path)

    for skip in SKIP_EXTS:
        if path.endswith(skip):
            if FLAGS.verbose:
                print("skipping due to extension: " + path)
            return True

    for skip in SKIP_PATHS:
        if str(pathlib.Path(path).resolve()).startswith(
            str(repo_abs_path.joinpath(skip).resolve())
        ):
            if FLAGS.verbose:
                print("skipping due to path prefix: " + path)
            return True

    with open(path, "r") as f:
        first_line = True
        line = None
        try:
            for fline in f:
                line = fline

                # Skip any '#!', '..', '<!--', '\*' or '{{/*' lines at the
                # start of the file
                if first_line:
                    first_line = False
                    if (
                        fline.startswith("#!")
                        or fline.startswith("..")
                        or fline.startswith("<!--")
                        or fline.startswith("/*")
                        or fline.startswith("{{/*")
                    ):
                        continue
                # Skip empty lines...
                if len(fline.strip()) != 0:
                    break
        except UnicodeDecodeError as ex:
            # If we get this exception on the first line then assume a
            # non-text file.
            if not first_line:
                raise ex
            if FLAGS.verbose:
                print("skipping binary file: " + path)
            return True

        if line is None:
            if FLAGS.verbose:
                print("skipping empty file: " + path)
            return True

        line = line.strip()

        # The next line must be the copyright line with a single year
        # or a year range. It is optionally allowed to have '# ' or
        # '// ' prefix.
        prefix = ""
        if line.startswith("# "):
            prefix = "# "
        elif line.startswith("// "):
            prefix = "// "
        elif line.startswith(".. "):
            prefix = ".. "
        elif not line.startswith(COPYRIGHT_YEAR_RE[0]):
            print(
                "incorrect prefix for copyright line, allowed prefixes '# ' or '// ', for "
                + path
                + ": "
                + line
            )
            return False

        # Check if the copyright year line matches the regex
        # and see if the year(s) are reasonable
        years = []

        copyright_row = line[len(prefix) :]
        if copyright_year_re.match(copyright_row):
            for year in (
                copyright_row.split(
                    "(c) " if "(c) " in copyright_row else "Copyright "
                )[1]
                .split(" NVIDIA ")[0]
                .split(",")
            ):
                if len(year) == 4:  # 2021
                    years.append(int(year))
                elif len(year) == 2:  # 21
                    years.append(int(year) + 2000)
                elif len(year) == 9:  # 2021-2022
                    years.append(int(year[0:4]))
                    years.append(int(year[5:9]))
                elif len(year) == 7:  # 2021-22
                    years.append(int(year[0:4]))
                    years.append(int(year[5:7]) + 2000)
                elif len(year) == 5:  # 21-23
                    years.append(int(year[0:2]) + 2000)
                    years.append(int(year[3:5]) + 2000)
        else:
            print("copyright year is not recognized for " + path + ": " + line)
            return False

        if years[0] > FLAGS.year:
            print(
                "copyright start year greater than current year for "
                + path
                + ": "
                + line
            )
            return False
        if years[-1] > FLAGS.year:
            print(
                "copyright end year greater than current year for " + path + ": " + line
            )
            return False
        for i in range(1, len(years)):
            if years[i - 1] >= years[i]:
                print("copyright years are not increasing for " + path + ": " + line)
                return False

        # Subsequent lines must match the copyright body.
        copyright_body = [
            l.rstrip() for i, l in enumerate(COPYRIGHT.splitlines()) if i > 0
        ]
        copyright_idx = 0
        for line in f:
            if copyright_idx >= len(copyright_body):
                break

            if len(prefix) == 0:
                line = line.rstrip()
            else:
                line = line.strip()

            if len(copyright_body[copyright_idx]) == 0:
                expected = prefix.strip()
            else:
                expected = prefix + copyright_body[copyright_idx]
            if line != expected:
                print("incorrect copyright body for " + path)
                print("  expected: '" + expected + "'")
                print("       got: '" + line + "'")
                return False
            copyright_idx += 1

        if copyright_idx != len(copyright_body):
            print(
                "missing "
                + str(len(copyright_body) - copyright_idx)
                + " lines of the copyright body"
            )
            return False

    if FLAGS.verbose:
        print("copyright correct for " + path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument("-y", "--year", type=int, required=True, help="Copyright year")
    parser.add_argument(
        "paths", type=str, nargs="*", default=None, help="Directories or files to check"
    )
    FLAGS = parser.parse_args()

    if FLAGS.paths is None or len(FLAGS.paths) == 0:
        parser.print_help()
        exit(1)

    ret = True
    for path in FLAGS.paths:
        if not os.path.isdir(path):
            if not visit(path):
                ret = False
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    if not visit(os.path.join(root, name)):
                        ret = False

    exit(0 if ret else 1)
