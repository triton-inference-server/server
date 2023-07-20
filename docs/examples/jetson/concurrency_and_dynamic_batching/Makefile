# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TARGET=people_detection
GCC=g++
GCC_PARMS+=-I../../server -I/usr/include/opencv4 -I../../core/include/ -I/usr/local/cuda/targets/aarch64-linux/include
GCC_PARMS+=-I${HOME}/tritonserver/include/tritonserver -D TRITON_ENABLE_GPU=ON -D TRITON_MIN_COMPUTE_CAPABILITY=5.3

GCC_LIBS=-L${HOME}/tritonserver/lib -L/usr/lib -L/usr/local/cuda/targets/aarch64-linux/lib
GCC_LIBS+=-lpthread -ltritonserver -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn -lcudart

all: $(TARGET)


%.o: %.cc
	$(GCC) $(GCC_PARMS) -c -g -o $@ $^

$(TARGET): $(TARGET).o
	$(GCC) $^ $(GCC_LIBS) -o $@

clean:
	rm -f $(TARGET).o $(TARGET)

.PHONY: all clean
