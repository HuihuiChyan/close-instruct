import os
import json
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora-model", type=str, default=None)
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--pool-manner", type=str, default="max_pool", choices=("max_pool", "avg_pool"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--overwrite-cache", action="store_true", default=False)
    parser.add_argument("--fetch-from-scratch", action="store_true", default=False)
    args = parser.parse_args()

    logging.info("loading data and model...")

    # load some data
    alapaca_eval_data = load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")["eval"]
    dataloader = torch.utils.data.DataLoader(alapaca_eval_data, batch_size=args.batch_size, shuffle=False)

    model_name = os.path.basename(os.path.normpath(args.model))

    print("Loading the base model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model)

    # add padding token if not already there
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    with torch.inference_mode():
        for samples in tqdm(dataloader):
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
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)