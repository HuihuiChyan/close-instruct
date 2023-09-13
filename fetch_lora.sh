WORK_DIR=/H1/huanghui

MODEL_PATH=/H1/models_huggingface/llama_7B
LORA_PATH=$WORK_DIR/open-instruct-output/huamnmix_lora

TRAIN_FILE=$WORK_DIR/open-instruct-data/human-mix/humanmix.data.jsonl
OUTPUT_FILE=$WORK_DIR/open-instruct-data/human-mix/humanmix.data.lora-embeds-8maxpool.jsonl

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    ./eval/fetch_lora_embeds.py \
    --model $MODEL_PATH \
    --lora-model $LORA_PATH \
    --train-file $TRAIN_FILE \
    --output-file $OUTPUT_FILE \
    --batch-size 32 \
    --max-seq-length 512 \
    --pool-manner "max_pool"

# OUTPUT_FILE=/mnt/bn/slp-llm/sft_huihui/open-instruct-data/sharegpt/sharegpt-den-nl.data.lora-embeds-8avgpool.jsonl

# accelerate launch --mixed_precision fp16 ./eval/fetch_lora_embeds.py \
#     --model $MODEL_PATH \
#     --lora-model $LORA_PATH \
#     --train-file $TRAIN_FILE \
#     --output-file $OUTPUT_FILE \
#     --batch-size 8 \
#     --pool-manner "avg_pool"