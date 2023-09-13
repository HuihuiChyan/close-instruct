OUTPUT_DIR=/mnt/bn/slp-llm/sft_huihui/open-instruct-output/output/sharegpt_all_lora
MERGE_PATH=/mnt/bn/slp-llm/sft_huihui/open-instruct-output/output/sharegpt_all_lora/merged
MODEL_PATH=/mnt/bn/slp-llm/sft_huihui/lmflow-instruct/llama-7b-hf

# python3 open_instruct/merge_lora.py \
#     --base_model_name_or_path $MODEL_PATH \
#     --lora_model_name_or_path $OUTPUT_DIR \
#     --output_dir $MERGE_PATH

python ./eval/alpaca_farm_eval.py \
    --model $MERGE_PATH \
    --save_folder $MERGE_PATH \
    --batch_size 8

python ./eval/mt_bench_eval.py \
    --model $MERGE_PATH \
    --save_folder $MERGE_PATH \
    --batch_size 8

OUTPUT_DIR="/mnt/bn/slp-llm/sft_huihui/open-instruct-output/output/sharegpt_ist_kmeans2k"

python ./eval/alpaca_farm_eval.py \
    --model $OUTPUT_DIR \
    --save_folder $OUTPUT_DIR \
    --batch_size 8

python ./eval/mt_bench_eval.py \
    --model $OUTPUT_DIR \
    --save_folder $OUTPUT_DIR \
    --batch_size 8

OUTPUT_DIR="/mnt/bn/slp-llm/sft_huihui/open-instruct-output/output/sharegpt_all"

python ./eval/alpaca_farm_eval.py \
    --model $OUTPUT_DIR \
    --save_folder $OUTPUT_DIR \
    --batch_size 8

python ./eval/mt_bench_eval.py \
    --model $OUTPUT_DIR \
    --save_folder $OUTPUT_DIR \
    --batch_size 8

# export TOKEN=$(cat '/etc/tce_dynamic/identity.token')
# python -m alpaca_eval.main evaluate \
#   --model_outputs $MERGE_PATH/sharegpt_ist_kmeans2k-alpaca-farm.json \
#   --annotators_config 'gpt4'

# LLMJUDGE_DIR="/mnt/bn/slp-llm/sft_huihui/open-instruct-data/llm_judge/"
# cp $MERGE_PATH/sharegpt_ist_kmeans2k-mt-bench.jsonl $LLMJUDGE_DIR/mt_bench/model_answer

# JUDGE_FILE=$LLMJUDGE_DIR/judge_prompts.jsonl
# python mt_bench_eval/gen_judgement.py \
#      --model-list sharegpt_ist_kmeans2k-mt-bench \
#      --data-dir $LLMJUDGE_DIR \
#      --judge-file $JUDGE_FILE\
#      --parallel 2