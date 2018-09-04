#!/bin/bash

MV2_USE_CUDA=1 srun --nodes=1 --ntasks-per-node=1 --mpibind=off --cpu_bind=mask_cpu:0x000001ff,0x0003fe00 -t 5 -p pvis \
../build/gnu.Release.pascal.llnl.gov/install/bin/lbann --num_gpus=1 \
--model=model.prototext \
--reader=data_reader_voc.prototext \
--optimizer=opt_sgd.prototext \
--mini_batch_size=1 \
--num_epochs=1000 2>&1 | tee ./debug.out
