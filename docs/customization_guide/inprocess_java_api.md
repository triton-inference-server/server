<!--
# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Java bindings for In-Process Triton Server API

The Triton Inference Server uses [Java CPP](https://github.com/bytedeco/javacpp)
to create bindings around Tritonserver to create Java API.

The API is documented in
[tritonserver.java](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/src/gen/java/org/bytedeco/tritonserver/global/tritonserver.java).
Alternatively, the user can refer to the web version [API docs](http://bytedeco.org/javacpp-presets/tritonserver/apidocs/)
generated from `tritonserver.java`.
**Note:** Currently, `tritonserver.java` contains bindings for both the `In-process C-API`
and the bindings for `C-API Wrapper`. More information about the [developer_tools/server C-API wrapper](https://github.com/triton-inference-server/developer_tools/blob/main/server/README.md) can be found in the [developer_tools repository](https://github.com/triton-inference-server/developer_tools/).

A simple example using the Java API can be found in
[Samples folder](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver/samples)
which includes `Simple.java` which is similar to
[`simple.cc`](https://github.com/triton-inference-server/server/blob/main/src/simple.cc).
Please refer to
[sample usage documentation](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver#sample-usage)
to learn about how to build and run `Simple.java`.

In the [QA folder](https://github.com/triton-inference-server/server/blob/main/qa), folders starting with L0_java include Java API tests.
These can be useful references for getting started, such as the
[ResNet50 test](https://github.com/triton-inference-server/server/blob/main/qa/L0_java_resnet).

## Java API setup instructions

To use the Tritonserver Java API, you will need to have the Tritonserver library
and dependencies installed in your environment. There are two ways to do this:

1. Use a Tritonserver docker container with
   1. `.jar` Java bindings to C API (recommended)
   2. maven and build bindings yourself
2. Build Triton from your environment without Docker (not recommended)

### Run Tritonserver container and install dependencies

To set up your environment with Triton Java API, please follow the following steps:
1. First run Docker container:
```
 $ docker run -it --gpus=all -v ${pwd}:/workspace nvcr.io/nvidia/tritonserver:<your container version>-py3 bash
```
2. Install `jdk`:
```bash
 $ apt update && apt install -y openjdk-11-jdk
```
3. Install `maven` (only if you want to build the bindings yourself):
```bash
$ cd /opt/tritonserver
 $ wget https://archive.apache.org/dist/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz
 $ tar zxvf apache-maven-3.8.4-bin.tar.gz
 $ export PATH=/opt/tritonserver/apache-maven-3.8.4/bin:$PATH
```

### Run Java program with Java bindings Jar

After ensuring that Tritonserver and dependencies are installed, you can run your
Java program with the Java bindings with the following steps:

1. Place Java bindings into your environment. You can do this by either:

   a. Building Java API bindings with provided build script:
      ```bash
      # Clone Triton client repo. Recommended client repo tag is: main
      $ git clone --single-branch --depth=1 -b <client repo tag>
                     https://github.com/triton-inference-server/client.git clientrepo
      # Run build script
      ## For In-Process C-API Java Bindings
      $ source clientrepo/src/java-api-bindings/scripts/install_dependencies_and_build.sh
      ## For C-API Wrapper (Triton with C++ bindings) Java Bindings
      $ source clientrepo/src/java-api-bindings/scripts/install_dependencies_and_build.sh --enable-developer-tools-server
      ```
      This will install the Java bindings to `/workspace/install/java-api-bindings/tritonserver-java-bindings.jar`

   *or*

   b. Copying "Uber Jar" from Triton SDK container to your environment
      ```bash
      $ id=$(docker run -dit nvcr.io/nvidia/tritonserver:<triton container version>-py3-sdk bash)
      $ docker cp ${id}:/workspace/install/java-api-bindings/tritonserver-java-bindings.jar <Uber Jar directory>/tritonserver-java-bindings.jar
      $ docker stop ${id}
      ```
      **Note:** `tritonserver-java-bindings.jar` only includes the `In-Process Java Bindings`. To use the `C-API Wrapper Java Bindings`, please use the build script.
2. Use the built "Uber Jar" that contains the Java bindings
   ```bash
   $ java -cp <Uber Jar directory>/tritonserver-java-bindings.jar <your Java program>
   ```

#### Build Java bindings and run Java program with Maven

If you want to make changes to the Java bindings, then you can use Maven to
build yourself. You can refer to part 1.a of [Run Java program with Java
bindings Jar](#run-java-program-with-java-bindings-jar) to also build the jar
yourself without any modifications to the Tritonserver bindings in
JavaCPP-presets.
You can do this using the following steps:

1. Create the JNI binaries in your local repository (`/root/.m2/repository`)
   with [`javacpp-presets/tritonserver`](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver).
   For C-API Wrapper Java bindings (Triton with C++ bindings), you need to
   install some build specific dependencies including cmake and rapidjson.
   Refer to [java installation script](https://github.com/triton-inference-server/client/blob/main/src/java-api-bindings/scripts/install_dependencies_and_build.sh)
   for dependencies you need to install and modifications you need to make for your container.
After installing dependencies, you can build the tritonserver project on javacpp-presets:
```bash
 $ git clone https://github.com/bytedeco/javacpp-presets.git
 $ cd javacpp-presets
 $ mvn clean install --projects .,tritonserver
 $ mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64
```
2. Create your custom `*.pom` file for Maven. Please refer to
   [samples/simple/pom.xml](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/samples/simple/pom.xml) as
   reference for how to create your pom file.
3. After creating your `pom.xml` file you can build your application with:
```bash
 $ mvn compile exec:java -Djavacpp.platform=linux-x86_64 -Dexec.args="<your input args>"
```