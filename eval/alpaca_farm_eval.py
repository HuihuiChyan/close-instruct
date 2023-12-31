import os
import json
import argparse
import logging

import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from utils import query_openai_chat_model, query_openai_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default=None)
parser.add_argument("--batch_size", "-b", type=int, default=8)
parser.add_argument("--openai_engine", "-o", type=str, default=None)
# where to save generations - default current directory
parser.add_argument("--output_file", "-s", type=str, default="")
args = parser.parse_args()

assert not (args.model and args.openai_engine), "only provide one of --model or --openai"
assert (args.model or args.openai_engine), "must provide one of --model or --openai"

logging.info("loading data and model...")
# load some data
# alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")["eval"]
alpaca_eval_data = datasets.load_dataset("/H1/huanghui/open-instruct-main/alpaca_eval/eval_data", "alpaca_farm_evaluation")["test"].select([i for i in range(20)])
dataloader = torch.utils.data.DataLoader(alpaca_eval_data, batch_size=args.batch_size, shuffle=False)

# use the data to get outputs for your model
if args.model is None:
    model_name = args.openai_engine
else:
    model_name = os.path.basename(os.path.normpath(args.model))

my_outputs = []
if args.openai_engine is None:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # add padding token if not already there
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    logging.info("model and data loaded!")
    logging.info("generating...")
    generation_config = GenerationConfig.from_pretrained(
        args.model,
        max_new_tokens=2048,
        temperature=0.0,
        # top_p=0.9,
        # do_sample=True,
        # num_return_sequences=1,
        # temperature=1.0,
        # top_k=0
    )
    model, accelerated_dataloader = accelerator.prepare(model, dataloader)
    with torch.inference_mode():
        for samples in tqdm(accelerated_dataloader):
            batch_outputs = []
            def convert_to_msg_format(input, instruction):
                if input == '':
                    input_text = "<|user|>\n" + instruction + "\n<|assistant|>\n"
                else:
                    prompt = instruction.strip() + "\n\n" + input.strip()
                    input_text = "<|user|>\n" + prompt + "\n<|assistant|>\n"
                return input_text
            inputs, instructions = samples['input'], samples['instruction']
            input_texts = [convert_to_msg_format(input, instruction) for input, instruction in zip(inputs, instructions)]
            input = tokenizer(input_texts, return_tensors="pt", padding="longest")
            input_ids = input.input_ids.to(model.device)
            attention_mask = input.attention_mask.to(model.device)
            outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
            for input_string, instruction, output in zip(samples['input'], samples['instruction'], outputs):
                output_string = tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=True)
                my_outputs.append({"instruction": instruction, "input": input_string, "generator": f"{model_name}-greedy-long", "output": output_string})
            print(my_outputs[-1])

        with open(args.output_file, 'w') as f:
            json.dump(my_outputs, f, indent=4)
else:
    completion_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    def convert_to_msg_format(input, instruction):
        if input == '':
            input_text = instruction
        else:
            input_text = instruction.strip() + "\n\n" + input.strip()
        return input_text
    for samples in tqdm(dataloader):
        inputs, instructions = samples['input'], samples['instruction']
        input_texts = [
            {"prompt": convert_to_msg_format(input, instruction), "id": "tmp" } for input, instruction in zip(inputs, instructions)
        ]
        openai_func = query_openai_model if args.openai_engine == "text-davinci-003" else query_openai_chat_model
        res = openai_func(
            args.openai_engine,
            input_texts,
            batch_size=args.batch_size,
            retry_limit=10000,  # since we will probably hit token limits kinda quickly.
            reuse_existing_outputs=True,
            **completion_kwargs
        )
        my_outputs += [{
            "instruction": instructions[i], "input": inputs[i], "generator": f"{model_name}-greedy-long", "output": r["output"]
        } for i, r in enumerate(res)]
        break
    with open(args.output_file, 'w') as f:
        json.dump(my_outputs, f, indent=4)