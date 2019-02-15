..
  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

Building
========

The TensorRT Inference Server is built using Docker and the TensorFlow
and PyTorch containers from `NVIDIA GPU Cloud (NGC)
<https://ngc.nvidia.com>`_. Before building you must install Docker
and nvidia-docker and login to the NGC registry by following the
instructions in :ref:`section-installing-prebuilt-containers`.

.. _section-building-the-server:

Building the Server
-------------------

To build a release version of the TensorRT Inference Server container,
change directory to the root of the repo and issue the following
command::

  $ docker build --pull -t tensorrtserver .

Incremental Builds
^^^^^^^^^^^^^^^^^^

For typical development you will want to run the *build* container
with your local repo’s source files mounted so that your local changes
can be incrementally built. This is done by first building the
*tensorrtserver_build* container::

  $ docker build --pull -t tensorrtserver_build --target trtserver_build .

By mounting /path/to/tensorrtserver/src into the container at
/workspace/src, changes to your local repo will be reflected in the
container::

  $ nvidia-docker run -it --rm -v/path/to/tensorrtserver/src:/workspace/src tensorrtserver_build

Within the container you can perform an incremental server build
with::

  # cd /workspace
  # bazel build -c opt --config=cuda src/servers/trtserver
  # cp /workspace/bazel-bin/src/servers/trtserver /opt/tensorrtserver/bin/trtserver

Similarly, within the container you can perform an incremental build
of the C++ and Python client libraries and example executables with::

  # cd /workspace
  # bazel build -c opt --config=cuda src/clients/…
  # mkdir -p /opt/tensorrtserver/bin
  # cp bazel-bin/src/clients/c++/image_client /opt/tensorrtserver/bin/.
  # cp bazel-bin/src/clients/c++/perf_client /opt/tensorrtserver/bin/.
  # cp bazel-bin/src/clients/c++/simple_client /opt/tensorrtserver/bin/.
  # mkdir -p /opt/tensorrtserver/lib
  # cp bazel-bin/src/clients/c++/librequest.so /opt/tensorrtserver/lib/.
  # cp bazel-bin/src/clients/c++/librequest.a /opt/tensorrtserver/lib/.
  # mkdir -p /opt/tensorrtserver/pip
  # bazel-bin/src/clients/python/build_pip /opt/tensorrtserver/pip/.

Some source changes seem to cause bazel to get confused and not
correctly rebuild all required sources. You can force bazel to rebuild
all of the inference server source without requiring a complete
rebuild of the TensorFlow and Caffe2 components by doing the following
before issuing the above build command::

  # rm -fr bazel-bin/src

.. include:: client.rst
   :start-after: build-client-begin-marker-do-not-remove
   :end-before: build-client-end-marker-do-not-remove

Building the Documentation
--------------------------

The inference server documentation is found in the docs/ directory and
is based on `Sphinx <http://www.sphinx-doc.org>`_. `Doxygen
<http://www.doxygen.org/>`_ integrated with `Exhale
<https://github.com/svenevs/exhale>`_ is used for C++ API
docuementation.

To build the docs install the required dependencies::

  $ apt-get update
  $ apt-get install -y --no-install-recommends doxygen
  $ pip install --upgrade sphinx sphinx-rtd-theme nbsphinx exhale

To get the Python client library API docs the TensorRT Inference
Server Python package must be installed::

  $ pip install --upgrade tensorrtserver-*.whl

Then use Sphinx to build the documentation into the build/html
directory::

  $ cd docs
  $ make clean html

To build the PDF version of documentation `LaTEX <https://www.tug.org/texlive/quickinstall.html>`
needs to be installed first. Additional requirements of python modules
can be met by simply running the following commands in docs directory::

  $ pip install -r requirements.txt

Once latex and python modules have been installed and updated single PDF for documentation
can be generated by running the following command. It will generate *NVIDIATRTIS.pdf* in *build/latex*
directory::

  $make clean latexpdf

Corrections and enhancements in Documentation are always welcome however it is advised that before
creating a pull request for the changes these are validated locally. A simple way to do it is to run web server
inside *docs/build/html* directory with following command and navigate through the modified documentation in the browser
at http://localhost:8000 ::
  $ python -m SimpleHTTPServer		# run inside docs/build/html
