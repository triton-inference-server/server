FROM centos:7.4.1708

# Install devtoolset 8
RUN yum install -y centos-release-scl && yum install -y devtoolset-8-gcc*

SHELL [ "/usr/bin/scl", "enable", "devtoolset-8"]

# Install basic tools
RUN yum -y install wget curl git which unzip bzip2 patch

# Install build tools
RUN yum -y install automake libtool mlocate openssl-devel bzip2-devel libffi-devel make perl-Data-Dumper

WORKDIR /workspace

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz
RUN tar zxvf cmake-3.*
WORKDIR /workspace/cmake-3.21.1
RUN ./bootstrap
RUN make
RUN make install
WORKDIR /workspace

# Install additional dependencies according to https://github.com/triton-inference-server/server/blob/main/build.py#L553
RUN yum -y install epel-release
RUN yum -y install boost-devel numactl-devel rapidjson-devel re2-devel

# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
# RUN yum install -y dnf
# RUN yum install -y dnf-data dnf-plugins-core libdnf-devel libdnf python2-dnf-plugin-migrate dnf-automatic
# RUN dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-11.0.3-1.x86_64.rpm
# RUN dnf install -y datacenter-gpu-manager

# Set up CMake compiler path
ENV CC=/opt/rh/devtoolset-8/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-8/root/usr/bin/c++

# Install libb64
RUN wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
RUN wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
RUN yum -y install libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
RUN yum -y install libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm

# Set up CMake file search path
ENV Protobuf_DIR=/tmp/citritonbuild/tritonserver/build/third-party/protobuf/lib64/cmake/protobuf
ENV prometheus-cpp_DIR=/tmp/citritonbuild/tritonserver/build/third-party/prometheus-cpp/lib64/cmake/prometheus-cpp

# Install OpenVINO
ARG ONNXRUNTIME_OPENVINO_VERSION=2021.2.200
ENV INTEL_OPENVINO_DIR /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}
ENV LD_LIBRARY_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH $INTEL_OPENVINO_DIR/tools:$PYTHONPATH
ENV IE_PLUGINS_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
ENV InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
ENV ngraph_DIR=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/cmake


RUN yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2021/setup/intel-openvino-2021.repo
RUN rpm --import https://yum.repos.intel.com/openvino/2021/setup/RPM-GPG-KEY-INTEL-OPENVINO-2021
RUN yum -y install yum-utils
RUN yum -y install intel-openvino-runtime-centos7-${ONNXRUNTIME_OPENVINO_VERSION}

# install python 3
RUN yum -y install python3

# Download and build ONNX runtime
WORKDIR /workspace
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_VERSION=1.9.0

RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
	(cd onnxruntime && git submodule update --init --recursive)
WORKDIR /workspace/onnxruntime
RUN rm -rf build
RUN rm -rf install
# RUN ./build.sh --config Release --build_shared_lib --use_openvino CPU_FP32 --skip_submodule_sync --parallel --build_dir /workspace/build
RUN ./build.sh --config Release --build_shared_lib --skip_submodule_sync --parallel --build_dir /workspace/build

# Copy Triton Server and onnxruntime_backend source code and cd into it
WORKDIR /workspace
COPY ./server server
COPY ./onnxruntime_backend onnxruntime_backend

# install latest boost 
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz
RUN tar -xzf boost_1_77_0.tar.gz
WORKDIR /workspace/boost_1_77_0
RUN ./bootstrap.sh --prefix=/opt/boost
RUN ./b2 install --prefix=/opt/boost --with=all; exit 0


# Build Triton Inference Server core
ARG tritonversion="main"
WORKDIR /workspace/server/
RUN git checkout ${tritonversion}
RUN python3 ./build.py \
    --cmake-dir=$(pwd)/build --build-dir=/tmp/citritonbuild --endpoint=http --endpoint=grpc --enable-logging --enable-stats --enable-tracing --enable-metrics \
    --backend=onnxruntime:${tritonversion} --repo-tag=common:${tritonversion} --repo-tag=core:${tritonversion} --repo-tag=backend:${tritonversion} --repo-tag=thirdparty:${tritonversion} --no-container-build
RUN cp -r /tmp/citritonbuild/opt/tritonserver /opt/

# Build ONNX runtime backend
RUN mkdir -p /opt/onnxruntime/lib && \
    cp /workspace/build/Release/libonnxruntime_providers_shared.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/Release/libonnxruntime.so.${ONNXRUNTIME_VERSION} \
       /opt/onnxruntime/lib && \
    (cd /opt/onnxruntime/lib && \
     ln -sf libonnxruntime.so.${ONNXRUNTIME_VERSION} libonnxruntime.so)

RUN mkdir -p /opt/onnxruntime/bin && \
    cp /workspace/build/Release/onnxruntime_perf_test \
       /opt/onnxruntime/bin && \
    cp /workspace/build/Release/onnx_test_runner \
       /opt/onnxruntime/bin && \
    (cd /opt/onnxruntime/bin && chmod a+x *)

RUN mkdir -p /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
       /opt/onnxruntime/include

WORKDIR /workspace/onnxruntime_backend/build
RUN cp /opt/intel/openvino_2021.2.200/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 /opt/onnxruntime/lib/
RUN cp /opt/intel/openvino_2021.2.200/deployment_tools/inference_engine/external/tbb/lib/libtbb.so /opt/onnxruntime/lib/
RUN cp /opt/intel/openvino_2021.2.200/deployment_tools/inference_engine/lib/intel64/libinference_engine.so /opt/onnxruntime/lib/
# Enable the line below if you want openVINO support
# RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON -DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=${ONNXRUNTIME_OPENVINO_VERSION} -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.8.0 -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF  -DTRITON_ONNXRUNTIME_LIB_PATHS=/opt/onnxruntime/lib -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/opt/onnxruntime/include  ..
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install -DTRITON_ENABLE_GPU=OFF -DTRITON_BUILD_ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION} -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF  -DTRITON_ONNXRUNTIME_LIB_PATHS=/opt/onnxruntime/lib -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/opt/onnxruntime/include  ..
RUN make install
RUN mkdir -p /opt/tritonserver/backends && \
    cp -r install/backends/onnxruntime /opt/tritonserver/backends/ && \
    cp /opt/onnxruntime/lib/* /opt/tritonserver/backends/onnxruntime/

# start the server
WORKDIR /opt/tritonserver