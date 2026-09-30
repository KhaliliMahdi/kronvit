cd "$(dirname "$0")/.."
GPU_NUM=${GPU_NUM:-2}
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}


start_time=$(date +"%s")
torchrun --nproc_per_node=$GPU_NUM \
     --master_port 29521\
     main.py  \
     --model deit_base_patch16_224 \
     --batch-size 128 \
     --data-set CIFAR \
     --data-path ${DATA_PATH:-./data/cifar100} \
     --output_dir ${OUTPUT_ROOT:-./output}/group_lasso_block4_2/ \
     --group_lasso \
     # --finetune ${OUTPUT_ROOT:-./output}/cifar100_train_common30/best_checkpoint.pth \

end_time=$(date +"%s")

# Calculate and echo the total time taken
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"