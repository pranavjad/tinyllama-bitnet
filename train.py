from transformers import (AutoTokenizer, AutoConfig, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from datasets import load_dataset
from huggingface_hub import login
import wandb
from utils import *

### Login
# Wandb is for logging and is optional.
hf_token = "<your_hf_token>"
wb_token = "<your_wb_token>"
wandb.login(key=wb_token)
login(token=hf_token)

### Load and tokenize training data. Uncomment these lines to load and tokenize yourself.
# data_source = "Skylion007/openwebtext"
# data = load_dataset(data_source)
# subset = load_dataset(data_source, split="train[:15%]")

# context_length = 256
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# def tokenize(element):
#     outputs = tokenizer(
#         element["text"],
#         truncation=False,
#         max_length=context_length,
#         return_overflowing_tokens=True,
#         return_length=True,
#     )
#     # Combine all tokens
#     combined = []
#     for tokenized_doc in outputs['input_ids']:
#         combined += tokenized_doc + [tokenizer.eos_token_id]
#     # Chunk
#     input_batch = []
#     for i in range(0, len(combined) - context_length, context_length):
#         input_batch.append(combined[i:i+context_length])
#     return {"input_ids": input_batch}

# tokenized_data = subset.map(
#     tokenize, batched=True, remove_columns=data["train"].column_names, 
# )
### End Load and tokenize training data

### Load pretokenized data (15% of openwebtext tokenized for ctx len of 256, ~1.5B tokens)
# You can subset this even further if you want a smaller dataset.

context_length = 256
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenized_data = load_dataset("xz56/openwebtext-tokenized-small")

total_tokens = tokenized_data['train'].num_rows * context_length
print(f"Training on {total_tokens:_} tokens")

### Adjust llama config to make the model tiny
config = AutoConfig.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

dim = 768
n_heads = 6
n_layers = 6
intermediate_size = 1536
config.hidden_size = dim
config.max_position_embeddings = dim
config.num_attention_heads = n_heads
config.num_hidden_layers = n_layers
config.num_key_value_heads = n_heads
config.intermediate_size = intermediate_size

### Create the llama model with our custom config. Convert it to bitnet.
# See utils.py for BitLinear and convert_to_bitnet function details.
model = LlamaForCausalLM(config)                  
convert_to_bitnet(model, copy_weights=False)

### Print number of parameters.
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")

### Set up DataCollator for creating batches
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

### Set up training arguments and begin training. Adjust these to your needs.
# Adjust the batch size until you can train on your device. Then increase accumulation steps to satisfy the following:
# tokens per batch = per_device_train_batch_size * gradient_accumulation_steps * 256
# Adjust this until tokens per batch is at least ~100k.
output_path = "<folder_to_save_checkpoints>"
args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=200,
    per_device_eval_batch_size=200,
    evaluation_strategy="steps",
    eval_steps=0.05,
    logging_steps=100,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    weight_decay=0.01,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    learning_rate=1.5e-3,
    save_steps=0.25,
    fp16=True,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
)

trainer.train()

trainer.save_model(f"{output_path}/final_model")
