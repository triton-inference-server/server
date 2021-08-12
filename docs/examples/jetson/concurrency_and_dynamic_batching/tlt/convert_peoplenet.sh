#!/bin/bash

./tlt-converter \
    -k tlt_encode \
    -d 3,544,960 \
    -i nchw \
    -t fp16 \
    -b 16 \
    -m 64 \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    -e ../trtis_model_repo_sample_1/peoplenet/1/model.plan \
    models/peoplenet/resnet34_peoplenet_pruned.etlt

cp ../trtis_model_repo_sample_1/peoplenet/1/model.plan ../trtis_model_repo_sample_2/peoplenet/1/model.plan

