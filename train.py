import torch
import torch.nn as nn
# import a config from transformers
from transformers import Trainer, TrainingArguments
# OpenAI GPT for text generation
from transformers import OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import DataCollatorForLanguageModeling
from process_data import *

# initialize a model from config
config = OpenAIGPTConfig(
    vocab_size=100000,
    n_positions=512,
    n_layer=6
)
model = False

# the pretrained tokenizer
tname = "Jojo_Tokenizer"
tokenizer = OpenAIGPTTokenizer.from_pretrained(tname)

# initialize a data collator 
data_collator = False # DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

# initialize dataset - process_data
dataset = False # CAN'T USE LINE BY LINE SO HOW

output = "output"

# initialize training arguments
training_args = TrainingArguments(
    output_dir="./"+output,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model("./"+output)