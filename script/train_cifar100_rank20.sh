#!/bin/bash

cd "$(dirname "$0")/.."
GPU_NUM=${GPU_NUM:-2}
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}


start_time=$(date +"%s") 

torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29508\
      main.py \
     --epochs 300 \
     --model kron_deit_tiny_patch16_224 \
     --batch-size 256 \
     --kron \
     --data-set CIFAR \
     --data-path ${DATA_PATH:-./data/cifar100} \
     --output_dir ${OUTPUT_ROOT:-./output}/cifar100_kron30/deit_tiny_patch16_224_20/ \
     --kron_rank 20\
     --shape_bias 3\
     ${FINETUNE:+--finetune "$FINETUNE"} \

end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"