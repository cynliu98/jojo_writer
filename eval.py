from transformers import pipeline

output = "jojo"

# allows model to fill in masked items
fill_mask = pipeline(
    "fill-mask",
    model="./"+output,
    tokenizer="./"+output
)