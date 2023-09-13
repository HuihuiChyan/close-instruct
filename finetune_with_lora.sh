cd /H1/huanghui/open-instruct-main/

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ninja
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn==2.1.1

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAIN_FILE="../open-instruct-data/human-mix/humanmix.data.jsonl"
OUTPUT_DIR="../open-instruct-output/huamnmix_lora"
MODEL_PATH="/H1/models_huggingface/llama_7B"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 8000 \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path $MODEL_PATH \
    --use_flash_attn \
    --use_lora \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name $MODEL_PATH \
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
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
