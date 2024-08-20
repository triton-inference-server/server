#!/bin/bash

### Helpers ###


# TODO: Detect vllm vs trtllm
function install_deps() {
    pushd openai/docker
    pip install -r requirements.txt
    pip install -r requirements_vllm.txt
    popd
}

function pre_test() {

    rm -rf openai/
    rm -f *.xml *.log

    # TODO: Use this instead when moving to devel container
    # cp -r ../../python/openai .
    cp -r /mnt/server/python/openai .

    install_deps
}

function run_test() {
    pushd openai/tests
    pytest -s -v --junitxml=test_openai.xml 2>&1 | tee test_openai.log
    cp *.xml *.log ../../
    popd
}

function post_test() {
    # no-op
}


### Test ###

pre_test
run_test
post_test
