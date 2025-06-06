#本文利用unsloth技术框架对Llama3大模型进行中文微调
#微调好的中文llama3下载链接 https://huggingface.co/leo009/dir/tree/main
#如有问题请联系up的徽信:stoeng

import PyPDF2
import re

#清洗PDF文档内容，包括章节抬头和出版信息等
def clean_extracted_text(text):
    """Clean and preprocess extracted text."""
    # Remove chapter titles and sections
    text = re.sub(r'^(Introduction|Chapter \d+:|What is|Examples:|Chapter \d+)', '', text, flags=re.MULTILINE)
    text = re.sub(r'ctitious', 'fictitious', text)
    text = re.sub(r'ISBN[- ]13: \d{13}', '', text)
    text = re.sub(r'ISBN[- ]10: \d{10}', '', text)
    text = re.sub(r'Library of Congress Control Number : \d+', '', text)
    text = re.sub(r'(\.|\?|\!)(\S)', r'\1 \2', text)  # Ensure space after punctuation
    text = re.sub(r'All rights reserved|Copyright \d{4}', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # Remove all newlines and replace newlines only after periods
    text = text.replace('\n', ' ')
    text = re.sub(r'(\.)(\s)', r'\1\n', text)

    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + ' '  # Append text of each page
    return text

def main():
    pdf_path = '/Users/charlesqin/Documents/The Art of Asking ChatGPT.pdf'  # Path to your PDF file
    extracted_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(extracted_text)

    # Output the cleaned text to a file
    with open('cleaned_text_output.txt', 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

if __name__ == '__main__':
    main()
#微调代码
from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset

file_path = "/home/Ubuntu/alpaca_gpt4_data_zh.json"


dataset = load_dataset("json", data_files={"train": file_path}, split="train")

dataset = dataset.map(formatting_prompts_func, batched = True,)




trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("dir", tokenizer, quantization_method = "f16")
#now you may locally host the output model in Ollama and play with it 
