export CUDA_VISIBLE_DEVICES=1,2

export OPENAI_PROXY='http://yazhedu:3wEYzA6Z6Sms3F5nfTjV@176.100.150.21:50100'
export OPENAI_API_KEY="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"
export OPENAI_ORGANIZATION="org-pamKQ62Lm9jl7JMDeQDpDXAR"

for FILE_NAME in "humanmix_random2k";
do
    MODEL_PATH="/H1/huanghui/open-instruct-output/$FILE_NAME/"
    ALPACA_FILE="/H1/huanghui/open-instruct-data/alpaca-farm/answers/$FILE_NAME-alpaca-farm.json"
    ALPACA_DIR="/H1/huanghui/open-instruct-data/alpaca-farm/"
    MTBENCH_FILE="/H1/huanghui/open-instruct-data/llm_judge/mt_bench/model_answer/$FILE_NAME-mt-bench.jsonl"

    # # batch_size has to be 10 for mt_bench
    # python3 ./eval/mt_bench_eval.py \
    #     --model $MODEL_PATH \
    #     --output_file $MTBENCH_FILE \
    #     --batch_size 10

    # python3 ./eval/alpaca_farm_eval.py \
    #     --model $MODEL_PATH \
    #     --output_file $ALPACA_FILE \
    #     --batch_size 8
  
    LLMJUDGE_DIR="/H1/huanghui/open-instruct-data/llm_judge/"
    JUDGE_FILE="/H1/huanghui/open-instruct-data/llm_judge/judge_prompts.jsonl"
    python -u mt_bench_eval/gen_judgement.py \
         --model-list $FILE_NAME-mt-bench \
         --data-dir $LLMJUDGE_DIR \
         --judge-file $JUDGE_FILE \
         --parallel 2

    # python3 mt_bench_eval/show_result.py \
    #     --input-file "/mnt/bn/slp-llm/sft_huihui/open-instruct-data/llm_judge/mt_bench/model_judgment/gpt-4_single.jsonl"

    # python3 -m alpaca_eval.main evaluate \
    #   --model_outputs $ALPACA_FILE \
    #   --output_path $ALPACA_DIR \
    #   --precomputed_leaderboard $ALPACA_DIR/leaderboard.csv \
    #   --annotators_config $ALPACA_DIR/config 

done