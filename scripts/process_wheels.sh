#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
set -e

# Usage: process_wheels.sh <DEPENDENCIES_DIR> <WHEELS_DIR> <OUTPUT_DIR>
DEPENDENCIES_DIR=$1
WHEELS_DIR=$2
OUTPUT_DIR=$3

if [[ -z "$DEPENDENCIES_DIR" || -z "$WHEELS_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <DEPENDENCIES_DIR> <WHEELS_DIR> <OUTPUT_DIR>"
    exit 1
fi

# Ensure output directory and subfolder for repaired wheels exist
mkdir -p "$OUTPUT_DIR/repaired"

for WHEEL_FILE in "$WHEELS_DIR"/*.whl; do
    echo "Processing wheel: $WHEEL_FILE..."

    # Extract the wheel into a temporary directory
    TEMP_DIR="$OUTPUT_DIR/temp"
    mkdir -p "$TEMP_DIR"
    unzip -q "$WHEEL_FILE" -d "$TEMP_DIR"

    # Add shared libraries to the `lib/` folder inside the wheel
    LIB_DIR="$TEMP_DIR/lib"
    mkdir -p "$LIB_DIR"
    echo "Copying shared libraries from $DEPENDENCIES_DIR to $LIB_DIR..."
    cp -u "$DEPENDENCIES_DIR"/* "$LIB_DIR"

    # Repackage the modified wheel
    REPACKED_WHEEL="${WHEEL_FILE%.whl}-repacked.whl"
    echo "Repacking the wheel..."
    pushd "$TEMP_DIR" > /dev/null
    zip -qr "$REPACKED_WHEEL" *
    popd > /dev/null

    rm -rf "${WHEELS_DIR}/tritonfrontend*"
    # Move the repacked wheel back to the wheels directory
    # mv -f "$REPACKED_WHEEL" "$WHEELS_DIR"

    # Repair the wheel using auditwheel_patched.py
    echo "Repairing the wheel to make it compliant with manylinux2014..."
    python3 -m auditwheel repair \
        --plat manylinux2014_x86_64 \
        --lib-sdir lib \
        "$WHEELS_DIR/$(basename "$REPACKED_WHEEL")" \
        -w "$OUTPUT_DIR/repaired"

    echo "Repaired wheel saved in $OUTPUT_DIR/repaired."

    # Clean up temporary directory
    rm -rf "$TEMP_DIR"
done

echo "All wheels processed and saved to $OUTPUT_DIR/repaired."
