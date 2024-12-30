#!/usr/bin/env python3
# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import platform
import subprocess
import sys

FLAGS = None


#### helper functions
def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print("<failed to log>", file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    print("error: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def fail_if(p, msg):
    if p:
        fail(msg)


def start_dockerfile(ddir, images, argmap, dockerfile_name, backends):
    # Set environment variables, set default user and install dependencies
    df = """
#
# Multistage build.
#
ARG TRITON_VERSION={}
ARG TRITON_CONTAINER_VERSION={}

FROM {} AS full
""".format(
        argmap["TRITON_VERSION"], argmap["TRITON_CONTAINER_VERSION"], images["full"]
    )

    # PyTorch, TensorFlow backends need extra CUDA and other
    # dependencies during runtime that are missing in the CPU-only base container.
    # These dependencies must be copied from the Triton Min image.
    if not FLAGS.enable_gpu and (
        ("pytorch" in backends)
        or ("tensorflow" in backends)
        or ("tensorflow2" in backends)
    ):
        df += """
FROM {} AS min_container

""".format(
            images["gpu-min"]
        )

    df += """
FROM {}

ENV PIP_BREAK_SYSTEM_PACKAGES=1
""".format(
        images["min"]
    )

    import build

    df += build.dockerfile_prepare_container_linux(
        argmap, backends, FLAGS.enable_gpu, platform.machine().lower()
    )
    # Copy over files
    df += """
WORKDIR /opt/tritonserver
COPY --chown=1000:1000 --from=full /opt/tritonserver/LICENSE .
COPY --chown=1000:1000 --from=full /opt/tritonserver/TRITON_VERSION .
COPY --chown=1000:1000 --from=full /opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf .
COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include include/
"""
    with open(os.path.join(ddir, dockerfile_name), "w") as dfile:
        dfile.write(df)


def add_requested_backends(ddir, dockerfile_name, backends):
    df = "# Copying over backends \n"
    for backend in backends:
        df += """COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/{} /opt/tritonserver/backends/{}
""".format(
            backend, backend
        )
    if len(backends) > 0:
        df += """
# Top-level /opt/tritonserver/backends not copied so need to explicitly set permissions here
RUN chown triton-server:triton-server /opt/tritonserver/backends
"""
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def add_requested_repoagents(ddir, dockerfile_name, repoagents):
    df = "#  Copying over repoagents \n"
    for ra in repoagents:
        df += """COPY --chown=1000:1000 --from=full /opt/tritonserver/repoagents/{} /opt/tritonserver/repoagents/{}
""".format(
            ra, ra
        )
    if len(repoagents) > 0:
        df += """
# Top-level /opt/tritonserver/repoagents not copied so need to explicitly set permissions here
RUN chown triton-server:triton-server /opt/tritonserver/repoagents
"""
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def add_requested_caches(ddir, dockerfile_name, caches):
    df = "#  Copying over caches \n"
    for cache in caches:
        df += """COPY --chown=1000:1000 --from=full /opt/tritonserver/caches/{} /opt/tritonserver/caches/{}
""".format(
            cache, cache
        )
    if len(caches) > 0:
        df += """
# Top-level /opt/tritonserver/caches not copied so need to explicitly set permissions here
RUN chown triton-server:triton-server /opt/tritonserver/caches
"""
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def end_dockerfile(ddir, dockerfile_name, argmap):
    # Install additional dependencies
    df = ""
    if argmap["SAGEMAKER_ENDPOINT"]:
        df += """
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
COPY --chown=1000:1000 --from=full /usr/bin/serve /usr/bin/.
"""
    with open(os.path.join(ddir, dockerfile_name), "a") as dfile:
        dfile.write(df)


def build_docker_image(ddir, dockerfile_name, container_name):
    # Create container with docker build
    p = subprocess.Popen(
        [
            "docker",
            "build",
            "-t",
            container_name,
            "-f",
            os.path.join(ddir, dockerfile_name),
            ".",
        ]
    )
    p.wait()
    fail_if(p.returncode != 0, "docker build {} failed".format(container_name))


def get_container_version_if_not_specified():
    if FLAGS.container_version is None:
        # Read from TRITON_VERSION file in server repo to determine version
        with open("TRITON_VERSION", "r") as vfile:
            version = vfile.readline().strip()
        import build

        _, FLAGS.container_version = build.container_versions(
            version, None, FLAGS.container_version
        )
        log("version {}".format(version))
    log("using container version {}".format(FLAGS.container_version))


def create_argmap(images, skip_pull):
    # Extract information from upstream build and create map other functions can
    # use
    full_docker_image = images["full"]
    min_docker_image = images["min"]
    enable_gpu = FLAGS.enable_gpu
    # Docker inspect environment variables
    base_run_args = ["docker", "inspect", "-f"]
    import re  # parse all PATH environment variables

    # first pull docker images
    if not skip_pull:
        log("pulling container:{}".format(full_docker_image))
        p = subprocess.run(["docker", "pull", full_docker_image])
        fail_if(
            p.returncode != 0,
            "docker pull container {} failed, {}".format(full_docker_image, p.stderr),
        )
    if enable_gpu:
        if not skip_pull:
            pm = subprocess.run(["docker", "pull", min_docker_image])
            fail_if(
                pm.returncode != 0 and not skip_pull,
                "docker pull container {} failed, {}".format(
                    min_docker_image, pm.stderr
                ),
            )
        pm_path = subprocess.run(
            base_run_args
            + [
                "{{range $index, $value := .Config.Env}}{{$value}} {{end}}",
                min_docker_image,
            ],
            capture_output=True,
            text=True,
        )
        fail_if(
            pm_path.returncode != 0,
            "docker inspect to find triton environment variables for min container failed, {}".format(
                pm_path.stderr
            ),
        )
        # min container needs to be GPU-support-enabled if the build is GPU build
        vars = pm_path.stdout
        e = re.search("CUDA_VERSION", vars)
        gpu_enabled = False if e is None else True
        fail_if(
            not gpu_enabled,
            "Composing container with gpu support enabled but min container provided does not have CUDA installed",
        )

    # Check full container environment variables
    p_path = subprocess.run(
        base_run_args
        + [
            "{{range $index, $value := .Config.Env}}{{$value}} {{end}}",
            full_docker_image,
        ],
        capture_output=True,
        text=True,
    )
    fail_if(
        p_path.returncode != 0,
        "docker inspect to find environment variables for full container failed, {}".format(
            p_path.stderr
        ),
    )
    vars = p_path.stdout
    log_verbose("inspect args: {}".format(vars))

    e0 = re.search("TRITON_SERVER_GPU_ENABLED=([\S]{1,}) ", vars)
    e1 = re.search("CUDA_VERSION", vars)
    gpu_enabled = False
    if e0 != None:
        gpu_enabled = e0.group(1) == "1"
    elif e1 != None:
        gpu_enabled = True
    fail_if(
        gpu_enabled != enable_gpu,
        "Error: full container provided was build with "
        "'TRITON_SERVER_GPU_ENABLED' as {} and you are composing container"
        "with 'TRITON_SERVER_GPU_ENABLED' as {}".format(gpu_enabled, enable_gpu),
    )
    e = re.search("TRITON_SERVER_VERSION=([\S]{6,}) ", vars)
    version = "" if e is None else e.group(1)
    fail_if(
        len(version) == 0,
        "docker inspect to find triton server version failed, {}".format(p_path.stderr),
    )
    e = re.search("NVIDIA_TRITON_SERVER_VERSION=([\S]{5,}) ", vars)
    container_version = "" if e is None else e.group(1)
    fail_if(
        len(container_version) == 0,
        "docker inspect to find triton container version failed, {}".format(vars),
    )
    dcgm_ver = re.search("DCGM_VERSION=([\S]{4,}) ", vars)
    dcgm_version = ""
    if dcgm_ver is None:
        dcgm_version = "2.2.3"
        log(
            "WARNING: DCGM version not found from image, installing the earlierst version {}".format(
                dcgm_version
            )
        )
    else:
        dcgm_version = dcgm_ver.group(1)
    fail_if(
        len(dcgm_version) == 0,
        "docker inspect to find DCGM version failed, {}".format(vars),
    )

    p_sha = subprocess.run(
        base_run_args
        + ['{{ index .Config.Labels "com.nvidia.build.ref"}}', full_docker_image],
        capture_output=True,
        text=True,
    )
    fail_if(
        p_sha.returncode != 0,
        "docker inspect of upstream docker image build sha failed, {}".format(
            p_sha.stderr
        ),
    )
    p_build = subprocess.run(
        base_run_args
        + ['{{ index .Config.Labels "com.nvidia.build.id"}}', full_docker_image],
        capture_output=True,
        text=True,
    )
    fail_if(
        p_build.returncode != 0,
        "docker inspect of upstream docker image build sha failed, {}".format(
            p_build.stderr
        ),
    )

    p_find = subprocess.run(
        ["docker", "run", full_docker_image, "bash", "-c", "ls /usr/bin/"],
        capture_output=True,
        text=True,
    )
    f = re.search("serve", p_find.stdout)
    fail_if(
        p_find.returncode != 0,
        "Cannot search for 'serve' in /usr/bin, {}".format(p_find.stderr),
    )
    argmap = {
        "NVIDIA_BUILD_REF": p_sha.stdout.rstrip(),
        "NVIDIA_BUILD_ID": p_build.stdout.rstrip(),
        "TRITON_VERSION": version,
        "TRITON_CONTAINER_VERSION": container_version,
        "DCGM_VERSION": dcgm_version,
        "SAGEMAKER_ENDPOINT": f is not None,
    }
    return argmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        required=False,
        help="Disable console output.",
    )
    group_qv.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=False,
        help='Name for the generated Docker image. Default is "tritonserver".',
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=False,
        help="Generated dockerfiles are placed here. Default to current directory.",
    )
    parser.add_argument(
        "--container-version",
        type=str,
        required=False,
        help="The version to use for the generated Docker image. If not specified "
        "the container version will be chosen automatically based on the "
        "repository branch.",
    )
    parser.add_argument(
        "--image",
        action="append",
        required=False,
        help="Use specified Docker image to generate Docker image. Specified as "
        '<image-name>,<full-image-name>. <image-name> can be "min", "gpu-min" '
        'or "full". Both "min" and "full" need to be specified at the same time.'
        'This will override "--container-version". "gpu-min" is needed for '
        "CPU-only container to copy TensorFlow and PyTorch deps.",
    )
    parser.add_argument(
        "--enable-gpu",
        nargs="?",
        type=lambda x: (str(x).lower() == "true"),
        const=True,
        default=True,
        required=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backend",
        action="append",
        required=False,
        help="Include <backend-name> in the generated Docker image. The flag may be "
        "specified multiple times.",
    )
    parser.add_argument(
        "--repoagent",
        action="append",
        required=False,
        help="Include <repoagent-name> in the generated Docker image. The flag may "
        "be specified multiple times.",
    )
    parser.add_argument(
        "--cache",
        action="append",
        required=False,
        help="Include <cache-name> in the generated Docker image. The flag may "
        "be specified multiple times.",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        required=False,
        help="Do not pull the required docker images. The user is responsible "
        "for pulling the upstream images needed to compose the image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=False,
        help="Only creates Dockerfile.compose, does not build the Docker image.",
    )

    FLAGS = parser.parse_args()

    if FLAGS.work_dir is None:
        FLAGS.work_dir = "."
    if FLAGS.output_name is None:
        FLAGS.output_name = "tritonserver"

    dockerfile_name = "Dockerfile.compose"

    if FLAGS.backend is None:
        FLAGS.backend = []
    if FLAGS.repoagent is None:
        FLAGS.repoagent = []
    if FLAGS.cache is None:
        FLAGS.cache = []

    # Initialize map of docker images.
    images = {}
    if FLAGS.image:
        for img in FLAGS.image:
            parts = img.split(",")
            fail_if(
                len(parts) != 2,
                "--image must specific <image-name>,<full-image-registry>",
            )
            fail_if(
                parts[0] not in ["min", "full", "gpu-min"],
                "unsupported image-name '{}' for --image".format(parts[0]),
            )
            log('image "{}": "{}"'.format(parts[0], parts[1]))
            images[parts[0]] = parts[1]
    else:
        get_container_version_if_not_specified()
        if FLAGS.enable_gpu:
            images = {
                "full": "nvcr.io/nvidia/tritonserver:{}-py3".format(
                    FLAGS.container_version
                ),
                "min": "nvcr.io/nvidia/tritonserver:{}-py3-min".format(
                    FLAGS.container_version
                ),
            }
        else:
            images = {
                "full": "nvcr.io/nvidia/tritonserver:{}-cpu-only-py3".format(
                    FLAGS.container_version
                ),
                "min": "ubuntu:22.04",
            }
    fail_if(len(images) < 2, "Need to specify both 'full' and 'min' images if at all")

    # For CPU-only image we need to copy some cuda libraries and dependencies
    # since we are using PyTorch, TensorFlow 1, TensorFlow 2 containers that
    # are not CPU-only.
    if (
        ("pytorch" in FLAGS.backend)
        or ("tensorflow" in FLAGS.backend)
        or ("tensorflow2" in FLAGS.backend)
    ) and ("gpu-min" not in images):
        images["gpu-min"] = "nvcr.io/nvidia/tritonserver:{}-py3-min".format(
            FLAGS.container_version
        )

    argmap = create_argmap(images, FLAGS.skip_pull)

    start_dockerfile(FLAGS.work_dir, images, argmap, dockerfile_name, FLAGS.backend)
    add_requested_backends(FLAGS.work_dir, dockerfile_name, FLAGS.backend)
    add_requested_repoagents(FLAGS.work_dir, dockerfile_name, FLAGS.repoagent)
    add_requested_caches(FLAGS.work_dir, dockerfile_name, FLAGS.cache)
    end_dockerfile(FLAGS.work_dir, dockerfile_name, argmap)

    if not FLAGS.dry_run:
        build_docker_image(FLAGS.work_dir, dockerfile_name, FLAGS.output_name)
