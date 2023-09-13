# Training Open Instruction-following Language Models

本仓库改编自 [open-instruct](https://github.com/allenai/open-instruct
), 原始的open-instruct包含如下内容：
1. 在多个指令数据上，全量或者lora形式的LLaMa指令微调；
2. 多个评测集上的验证脚本；
3. 还支持创建或者合并模型权重差值；

在原始的open-instruct基础上，我添加了下面的功能：
1. 调用公司的GPT4接口进行alpaca_eval或者mt_bench的评测；
2. 从训好的lora模型中取出一条数据对应的lora_embedding；

## 准备工作

按照下面的命令安装环境：

```bash
pip install ninja
pip install -r requirements.txt
pip install --no-build-isolation flash-attn
```

下载LLaMA模型并且完成权重合并。你也可以用我合并好的模型：/mnt/bn/slp-llm/sft_huihui/lmflow-instruct/llama-7b-hf

原始的open-instruct提供了大量的数据下载和预处理脚本，可以参考该脚本准备你的训练数据：./scripts/prepare_train_data.sh

还需要将alpaca_farm和mt_bench的配置文件拷贝到你个人存储盘的目录：/mnt/bn/slp-llm/sft_huihui/evaluation_dir
请务必执行这一句拷贝操作，并且将后续的evaluation结果和缓存都写入到拷贝后的目录中。请不要将evaluation结果写入到拷贝前的目录中。

## 模型训练

执行finetune.sh脚本进行全参数的指令微调，脚本的内容如下：

```bash
# 下面三行用于在merlin平台上配置环境
pip install ninja
pip install -r requirements.txt
pip install --no-build-isolation flash-attn

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAIN_FILE="/path/to/your/train_data_file"
OUTPUT_DIR="/path/to/your/model_output_dir"
MODEL_PATH="/path/to/your/llama_model_dir"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 8000 \
    --deepspeed_config_file ds_configs/stage2_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path $MODEL_PATH \
    --use_flash_attn \
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
    --num_train_epochs 5 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
```

执行finetune_with_lora.sh脚本进行lora的指令微调，脚本的内容如下：

```bash
# 下面三行用于在merlin平台上配置环境
pip install ninja
pip install -r requirements.txt
pip install --no-build-isolation flash-attn

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAIN_FILE="/path/to/your/train_data_file"
OUTPUT_DIR="/path/to/your/model_output_dir"
MODEL_PATH="/path/to/your/llama_model_dir"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 8000 \
    --deepspeed_config_file ds_configs/stage2_no_offloading_accelerate.conf \
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
    --num_train_epochs 5 \
    --output_dir $OUTPUT_DIR \
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
```

注意上述脚本都是在四卡A100的环境下进行，使用的是lab_pytorch2的官方镜像。并且由于deepspeed zero3会出一个奇怪的BUG，我用的都是zero2的配置。


## 模型验证

执行下面的脚本，自动从huggingface datasets中加载数据集进行验证。

### alpaca_eval

```bash
export TOKEN=$(cat '/etc/tce_dynamic/identity.token')
MODEL_NAME="your_model_name"
MODEL_PATH="/path/to/your/model_output_dir"
EVAL_DIR="/path/to/your/evaluation_dir"
ALPACA_FILE="$EVAL_DIR/alpaca-farm/answers/$FILE_NAME-alpaca-farm.json"

# 这一条python命令需要在a100的环境下执行
python3 ./eval/alpaca_farm_eval.py \
    --model $MODEL_PATH \
    --output_file $ALPACA_FILE \
    --batch_size 8

# 这一条python命令不需要在GPU环境下执行
python3 -m alpaca_eval.main evaluate \
  --model_outputs $ALPACA_FILE \
  --output_path $EVAL_DIR/alpaca-farm \
  --precomputed_leaderboard $EVAL_DIR/alpaca-farm/leaderboard.csv \
  --annotators_config $EVAL_DIR/alpaca-farm/config 
```

### mt_bench_eval
```bash
export TOKEN=$(cat '/etc/tce_dynamic/identity.token')
MODEL_NAME="your_model_name"
MODEL_PATH="/path/to/your/model_output_dir"
EVAL_DIR="/path/to/your/evaluation_dir"
MTBENCH_FILE="$EVAL_DIR/llm_judge/mt_bench/model_answer/$FILE_NAME-mt-bench.jsonl"

# 这一条python命令需要在a100的环境下执行
# 由于温度系数的原因，这里的batch_size必须是10
python3 ./eval/mt_bench_eval.py \
    --model $MODEL_PATH \
    --output_file $MTBENCH_FILE \
    --batch_size 10

# 这一条python命令不需要在GPU环境下执行
python mt_bench_eval/gen_judgement.py \
      --model-list $FILE_NAME-mt-bench \
      --data-dir "$EVAL_DIR/llm_judge" \
      --judge-file "$EVAL_DIR/llm_judge/judge_prompts.jsonl" \
      --parallel 2

# 这一条python命令不需要在GPU环境下执行
python3 mt_bench_eval/show_result.py \
      --input-file "$EVAL_DIR/llm_judge/mt_bench/model_judgment/gpt-4_single.jsonl"
```

## LoRA嵌入
在训练好lora模型之后，你可以执行下面的脚本，提取每一条数据对应的lora嵌入。
注意这里的INFER_FILE要处理成和训练集相同的格式，并且lora模型不需要合并。

```bash
MODEL_PATH="/path/to/original/llama_model_dir"
LORA_PATH="/path/to/your/lora_model_dir"
INFER_FILE="/path/to/your/infer_file_path"
OUTPUT_FILE="/path/to/your/output_file_path"

accelerate launch --mixed_precision bf16 ./eval/fetch_lora_embeds.py \
    --model $MODEL_PATH \
    --lora-model $LORA_PATH \
    --train-file $INFER_FILE \
    --output-file $OUTPUT_FILE \
    --batch-size 32 \
    --max-seq-length 512 \
    --pool-manner "max_pool"
```

## 引用

如果你觉得该仓库有用，你也不需要引用任何的论文或者链接，如果你实在过意不去，那你可以往这个支付宝账号发个红包：2745580384@qq.com