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
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from modeling_llama_lora import LlamaLoraForCausalLM

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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--pool-manner", type=str, default="max_pool", choices=("max_pool", "avg_pool"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--overwrite-cache", action="store_true", default=False)
    parser.add_argument("--fetch-from-scratch", action="store_true", default=False)
    args = parser.parse_args()

    accelerator = Accelerator()

    logging.info("loading data and model...")

    model_name = os.path.basename(os.path.normpath(args.model))

    peft_config = PeftConfig.from_pretrained(args.lora_model)
    print("Loading the base model...")
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    base_model = LlamaLoraForCausalLM.from_pretrained(
        args.model if args.model else peft_config.model, device_map=device_map, 
    ) 
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model)
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model)

    embedding_size = lora_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model.")
        print("Resizing the token embeddings of the merged model...")
        lora_model.resize_token_embeddings(len(tokenizer))

    data_files = {"train": args.train_file}
    raw_datasets = load_dataset("json", data_files=data_files,)

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    # lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]

    if not args.fetch_from_scratch and os.path.exists(args.output_file):
        lines = open(args.output_file, "r").readlines()
        train_dataset = train_dataset.select(np.arange(len(train_dataset))[len(lines):])

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=lora_model, padding="longest"),
        batch_size=args.batch_size,
    )

    model = lora_model

    lora_filename = f"{args.output_file}"

    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    with torch.inference_mode(), open(args.output_file, "a+") as fout:
        for samples in tqdm(train_dataloader):
            
            input_ids = samples.input_ids
            attention_mask = samples.attention_mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_lora=True)
            
            # query_embeds = torch.stack(outputs[0], dim=1)
            # value_embeds = torch.stack(outputs[1], dim=1)
            # lora_embeds = outputs[1][-4] # select -4 layer as representation
            lora_embeds = torch.stack(outputs[1][-8:], dim=1).mean(dim=1)

            lora_embeds = accelerator.gather(lora_embeds)
            attention_mask = accelerator.gather(attention_mask)

            if args.pool_manner == "max_pool":                
                zero_matrix = torch.zeros_like(attention_mask)
                ninf_matrix = torch.zeros_like(attention_mask) + float('-inf')

                lora_embeds_maxpool = (lora_embeds + torch.where(attention_mask==0, ninf_matrix, zero_matrix).unsqueeze(-1)).max(dim=-2).values

                if accelerator.is_main_process:
                    for lora_embed in lora_embeds_maxpool:
                        fout.write(str(lora_embed.tolist())+"\n")
            
            elif args.pool_manner == "avg_pool":
                lora_embeds_avgpool = (lora_embeds * attention_mask.unsqueeze(-1)).mean(dim=-2)

                if accelerator.is_main_process:
                    for lora_embed in lora_embeds_avgpool:
                        fout.write(str(lora_embed.tolist())+"\n")            
                
    # lora_query_maxpool_filename = f"{args.output_file}.lora-query-embeds-maxpool.npy"
    # lora_value_maxpool_filename = f"{args.output_file}.lora-value-embeds-maxpool.npy"
    # lora_query_avgpool_filename = f"{args.output_file}.lora-query-embeds-avgpool.npy"
    # lora_value_avgpool_filename = f"{args.output_file}.lora-value-embeds-avgpool.npy"

    # all_lora_query_states_maxpool = []
    # all_lora_value_states_maxpool = []
    # all_lora_query_states_avgpool = []
    # all_lora_value_states_avgpool = []

    # with torch.inference_mode():
    #     for samples in tqdm(train_dataloader):
            
    #         input_ids = samples.input_ids.to(model.device)
    #         attention_mask = samples.attention_mask.to(model.device)
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_lora=True)
            
    #         lora_query_states = torch.cat(outputs[-2], dim=-1)
    #         lora_value_states = torch.cat(outputs[-1], dim=-1)
            
    #         zero_matrix = torch.zeros_like(attention_mask)
    #         ninf_matrix = torch.zeros_like(attention_mask) + float('-inf')

    #         lora_query_states_maxpool = (lora_query_states + torch.where(attention_mask==0, ninf_matrix, zero_matrix).unsqueeze(-1)).max(dim=-2).values
    #         lora_value_states_maxpool = (lora_value_states + torch.where(attention_mask==0, ninf_matrix, zero_matrix).unsqueeze(-1)).max(dim=-2).values

    #         lora_query_states_avgpool = (lora_query_states * attention_mask.unsqueeze(-1)).mean(dim=-2)
    #         lora_value_states_avgpool = (lora_value_states * attention_mask.unsqueeze(-1)).mean(dim=-2)

    #         all_lora_query_states_maxpool.extend(lora_query_states_maxpool.tolist())
    #         all_lora_value_states_maxpool.extend(lora_value_states_maxpool.tolist())
    #         all_lora_query_states_avgpool.extend(lora_query_states_avgpool.tolist())
    #         all_lora_value_states_avgpool.extend(lora_value_states_avgpool.tolist())

    # with open(os.path.join(args.save_folder, lora_query_avgpool_filename), 'w') as fquery:
    #     np.save(fquery, np.array(all_lora_query_states_avgpool))

    # with open(os.path.join(args.save_folder, lora_query_avgpool_filename), 'w') as fvalue:
    #     np.save(fvalue, np.array(all_lora_value_states_avgpool))

    # with open(os.path.join(args.save_folder, lora_query_maxpool_filename), 'w') as fquery:
    #     np.save(fquery, np.array(all_lora_query_states_maxpool))

    # with open(os.path.join(args.save_folder, lora_query_maxpool_filename), 'w') as fvalue:
    #     np.save(fvalue, np.array(all_lora_value_states_maxpool))