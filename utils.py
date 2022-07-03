import json
from config import *
from torchtext.vocab import build_vocab_from_iterator


def load_atis_test(filepath):
	atis_data = json.load(open(filepath))
	atis_data = atis_data['rasa_nlu_data']
	common_examples = atis_data['common_examples']
	regex_features = atis_data['regex_features']
	lookup_tables = atis_data['lookup_tables']
	entity_synonyms = atis_data['entity_synonyms']
	return atis_data

def load_atis_train(filepath):
	atis_data = json.load(open(filepath))
	atis_data = atis_data['rasa_nlu_data']
	return atis_data

def build_vocab(text, vocab):
	# add words in text to vocab

	return vocab

if __name__ == "__main__":
	print('#' * 10, " ATIS ", '#' * 10) 

	atis_train = load_atis_train(ATIS_TRAIN_FILE)
	atis_test = load_atis_test(ATIS_TEST_FILE)
	print('ATIS Train samples:', len(atis_train['common_examples']))
	print('ATIS Test samples:', len(atis_test['common_examples']))

	#intent_vocab = []
	text = []
	for i in range(len(atis_train['common_examples'])):
		#intent_vocab.append(atis_train['common_examples'][i]['intent'])
		text.append(atis_train['common_examples'][i]['text'])
		#for k in atis_train['common_examples'][i].keys():
		#	print(k, ' ', atis_train['common_examples'][i][k])
	#intent_vocab = set(intent_vocab)
	#text = set(text_vocab)
	#print(text_vocab)
	#print(intent_vocab)

	def yield_text_file_tokens(file_path):
		with io.open(file_path, encoding = 'utf-8') as f:
			for line in f:
				yield line.strip().split()

	def yield_text_array_tokens(array):
		for line in array:
			line = line.strip().split()
			yield line

	vocab = build_vocab_from_iterator(yield_text_array_tokens(text), specials=["<unk>"])
	print(vocab.get_itos())