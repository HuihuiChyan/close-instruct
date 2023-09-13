import os
import json
import argparse
import logging
from random import sample

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
# mt_bench_data = datasets.load_dataset("dim/mt_bench_en")["train"]
mt_bench_data = datasets.load_dataset("/H1/huanghui/open-instruct-main/mt_bench_eval/eval_data")["train"]
dataloader = torch.utils.data.DataLoader(mt_bench_data, batch_size=args.batch_size, shuffle=False)
# use the data to get outputs for your model
if args.model is None:
    model_name = args.openai_engine
else:
    model_name = os.path.basename(os.path.normpath(args.model))

my_outputs = []
if args.openai_engine is None:
    temperature_config = {
        "writing": 0.7,
        "roleplay": 0.7,
        "extraction": 0.0,
        "math": 0.0,
        "coding": 0.0,
        "reasoning": 0.0,
        "stem": 0.1,
        "humanities": 0.1,
    }
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
        max_new_tokens=1024,
        # top_p=0.9,
        # do_sample=True,
        # num_return_sequences=1,
        # temperature=1.0,
        # top_k=0
    )
    with torch.inference_mode():
        for samples in tqdm(dataloader):
            temperature = temperature_config[samples["category"][0]]
            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True
            def convert_to_msg_format(input, instruction):
                if input == '':
                    input_text = "<|user|>\n" + instruction + "\n<|assistant|>\n"
                else:
                    prompt = instruction.strip() + "\n\n" + input.strip()
                    input_text = "<|user|>\n" + prompt + "\n<|assistant|>\n"
                return input_text
            input_texts = ["<|user|>\n" + turns + "\n<|assistant|>\n" for turns in samples['turns'][0]]
            input = tokenizer(input_texts, return_tensors="pt", padding="longest")
            input_ids = input.input_ids.to(model.device)
            attention_mask = input.attention_mask.to(model.device)
            outputs = model.generate(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     generation_config=generation_config,
                                     temperature=temperature,
                                     do_sample=do_sample)
            outputs = [tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=True) for output in outputs]

            second_input_texts = [i+o+"<|user|>\n" + t + "\n<|assistant|>\n" for i,o,t in zip(input_texts, outputs, samples['turns'][1])]
            input = tokenizer(second_input_texts, return_tensors="pt", padding="longest")
            input_ids = input.input_ids.to(model.device)
            attention_mask = input.attention_mask.to(model.device)
            second_outputs = model.generate(input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            generation_config=generation_config,
                                            temperature=temperature,
                                            do_sample=do_sample)
            second_outputs = [tokenizer.decode(output[input_ids.size(1):], skip_special_tokens=True) for output in second_outputs]
            for question_id, output, second_output in zip(samples["question_id"].tolist(), outputs, second_outputs):
                my_outputs.append({"question_id": question_id, "answer_id": "answer_id", "model_id": "model_id", "choices": [{"index": 0, "turns": [output, second_output]}]})
            print(my_outputs[-1])
    with open(args.output_file, 'w') as f:
        for output in my_outputs:
            f.write(json.dumps(output, ensure_ascii=False)+"\n")
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
    with open(args.output_file, 'w') as f:
        json.dump(my_outputs, f, indent=4)


print(my_outputs[0])