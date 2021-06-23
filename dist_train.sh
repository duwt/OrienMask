#!/usr/bin/env bash

GPUIDS=${CUDA_VISIBLE_DEVICES//,/}
GPUS=${#GPUIDS}
PORT=${MASTER_PORT:-29500}

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
                            --nproc_per_node=${GPUS} \
                            --master_port=${PORT} \
                            train.py ${@:1}
