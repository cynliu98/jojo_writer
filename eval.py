import argparse

from transformers import pipeline
from transformers import OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

output = "jojo"
MAX_LENGTH = int(10000)
device = "cpu" # change to CUDA/GPU later

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

# allows model to fill in masked items
# what the heck is this actually lol
fill_mask = pipeline(
    "fill-mask",
    model="./"+output,
    tokenizer="./"+output
)
# end "what the heck is this" section

# SOURCE: https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
# ---------------------------- Parser ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
)

parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--length", type=int, default=20)
parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)

parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

set_seed(args)

# ---------------------------- Set up model and tokenizer ----------------------------
tname = "Jojo_Tokenizer"
tokenizer = OpenAIGPTTokenizer.from_pretrained(tname)
model = OpenAIGPTLMHeadModel.from_pretrained("./"+output) # add the actual path after it's trained
args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

prompt_text = input("Prompt for generation? (Write ~1-2 sentences in the stye of a script, or nothing to exit): ")
prompt_text = prompt_text.strip()

# ---------------------------- Generate text ----------------------------
while prompt_text:
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    prompt_text = input("Prompt for generation? (Write ~1-2 sentences in the stye of a script, or nothing to exit): ")
    prompt_text = prompt_text.strip()