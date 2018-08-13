#!/bin/bash

MV2_USE_CUDA=1 srun --nodes=1 --ntasks-per-node=2 --mpibind=off --cpu_bind=mask_cpu:0x000001ff,0x0003fe00 -t 1440 \
../build/gnu.Release.pascal.llnl.gov/install/bin/lbann --num_gpus=2 \
--model=model_resnet50.prototext \
--reader=data_reader_voc.prototext \
--optimizer=../model_zoo/optimizers/opt_sgd.prototext \
--mini_batch_size=1 \
--num_epochs=100 2>&1 | tee ./output.txt
