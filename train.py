import torch
import torch.nn as nn
# import a config from transformers
from transformers import Trainer, TrainingArguments
from process_data import *

# initialize a model from a transformers config
model = False

# initialize a data collator WHAT IS THIS
data_collator = False

# initialize dataset - process_data
dataset = False

output = "jojo"

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