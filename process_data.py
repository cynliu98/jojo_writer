import re
from tokenizers import BertWordPieceTokenizer as WordTokenizer

# reformat the data and train a custom tokenizer on it
# try Word-based tokenizer first

# clean the subtitle files into a more readable format
# saves result as a new file associated with same episode
def clean_files(fname):
	pass

# combines cleaned files into corpuses for training, evaluating, testing
# saves results
def make_corpus(fnames):
	pass

# given paths to cleaned text files, trains a tokenizer and saves it
def train_tokenizer(tname,
					files,
					vocab_size=52_000,
					min_frequency=2,
					special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"]):
	pass
