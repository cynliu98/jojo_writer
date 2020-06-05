import re
from tokenizers import BertWordPieceTokenizer as WordTokenizer

# reformat the data and train a custom tokenizer on it
# try Word-based tokenizer first

FORMAT_INDEX = 0
STYLE_INDEX = 4
SPEAKER_INDEX = 5
TEXT_INDEX = -1

# clean the subtitle .ass files into a more readable format
# saves result as a new file associated with same episode
def wipe_ass(fname):
	pre_title_lines = []
	with open(fname+'.ass', 'r') as rf, open(fname+'_clean.ass', 'w') as wf:
		at_dialogue = False
		reached_title = False
		for line in rf:
			if not(at_dialogue) and not("[Events]" in line):
				continue
			if not(at_dialogue): # hit the "[Events]" line
				at_dialogue = True
				continue 

			# process the dialogue and write it to file in a cleaner method
			tokens = split_line(line)
			if tokens[FORMAT_INDEX] == "Dialogue":
				speaker = tokens[SPEAKER_INDEX]
				text = tokens[TEXT_INDEX]
				style = tokens[STYLE_INDEX]

				if not(reached_title):
					clean_str = speaker
					if style == "Thoughts" or style == "Italic":
						clean_str += " (thinking)"
					clean_str += ": " + text
					pre_title_lines.append(clean_str)
				else:
					continue # TODO

				# only start writing after we see the Title, so Title is at beginning
				if "Title " in style and "\\pos" in text:
					reached_title = True

# correctly splits lines in .ass files
# Returns: the split line
def split_line(line):
	raw_tokens = line.split(',')
	# parse the first part
	format_layer = raw_tokens[0].split(':')
	format_type = format_layer[0]; layer = format_layer[1]

	# combine dialogue parts separated by commas
	text = ','.join(raw_tokens[9:])
	return [format_type, layer] + raw_tokens[1:9] + [text]

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
