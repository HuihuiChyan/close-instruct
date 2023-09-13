sudo chown -R tiger /mnt/bn/slp-llm/sft_huihui/open-instruct-output/

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAIN_FILE="/mnt/bn/slp-llm/sft_huihui/open-instruct-data/sharegpt/sharegpt.data.jsonl"
OUTPUT_DIR="/mnt/bn/slp-llm/sft_huihui/open-instruct-output/output/sharegpt_all"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 8000 \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /mnt/bn/slp-llm/sft_huihui/lmflow-instruct/llama-7b-hf \
    --use_flash_attn \
    --tokenizer_name /mnt/bn/slp-llm/sft_huihui/lmflow-instruct/llama-7b-hf \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1