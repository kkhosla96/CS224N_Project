import pickle
import re

def pad_terms(terms, pad_token, max_term_length):
	'''
	Note: taken from Kush's implementation in assignment 4 for 224N 
	Pad list of sentences according to the longest sentence in the batch.
	@param sents (List[List[int]]): list of sentences, where each sentence
	is represented as a list of words
	@param pad_token (int): padding token
	@returns sents_padded (List[List[int]]): list of sentences where terms shorter
	than the max length sentence are padded out with the pad_token, such that
	each terms in the batch now has equal length.
	'''
	terms_padded = []

	for term in terms:
		diff = max_term_length - len(term)
		terms_padded.append((term + [pad_token] * diff)[:max_term_length])

	return terms_padded

def precision(predicted_terms, gold):
	return len(predicted_terms & gold) / len(predicted_terms) if len(predicted_terms) > 0 else 0

def recall(predicted_terms, gold):
	return len(predicted_terms & gold) / len(gold) if len(predicted_terms) > 0 else 0

def calculate_precision_and_recall(labeled_file, label_file, seed_file, gold_file):
	labeled, labels = get_labeled_and_labels(labeled_file, label_file)
	positive_seed_set = get_positive_seed_set(seed_file)
	golds = get_gold_terms(gold_file)
	predicted_positive = set([' '.join(labeled[i]) for i in range(len(labels)) if labels[i]])
	golds = set(golds)
	predicted_positive -= positive_seed_set
	golds -= positive_seed_set
	p = precision(predicted_positive, golds)
	r = recall(predicted_positive, golds)
	return (p, r)

def get_labeled_and_labels(labeled_file, label_file):
	with open(labeled_file, 'rb') as f:
		labeled = pickle.load(f)
	with open(label_file, 'rb') as f:
		labels = pickle.load(f)
	return labeled, labels

def get_positive_seed_set(seed_file):
	ret = set()
	with open(seed_file) as f:
		# remove the \n
		line = f.readline()[:-1]
		while line:
			split = line.split()
			if int(line[-1]) == 1:
				ret.add(' '.join(split[:-1]))
			line = f.readline()[:-1]
	return ret

def get_gold_terms(gold_file):
	with open(gold_file, 'rb') as f:
		return set(pickle.load(f))

def text_to_pickle(text_file, pickle_file):
	output_set = set()
	with open(text_file, "r") as text:
		for line in text:
			output_set.add(line.strip())
	with open(pickle_file, "wb") as f:
		pickle.dump(output_set, f)

# the following function is from
# https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings
def normalize(s):
	"""
	Given a text, cleans and normalizes it. Feel free to add your own stuff.
	"""
	s = s.lower()
	# Replace ips
	s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
	# Isolate punctuation
	s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
	# Remove some special characters
	s = re.sub(r'([\;\:\|•«\n])', ' ', s)
	#squash spaces
	s = re.sub(' +', ' ', s)
	# Replace numbers and symbols with language
	s = s.replace('&', ' and ')
	s = s.replace('@', ' at ')
	s = s.replace('0', ' zero ')
	s = s.replace('1', ' one ')
	s = s.replace('2', ' two ')
	s = s.replace('3', ' three ')
	s = s.replace('4', ' four ')
	s = s.replace('5', ' five ')
	s = s.replace('6', ' six ')
	s = s.replace('7', ' seven ')
	s = s.replace('8', ' eight ')
	s = s.replace('9', ' nine ')

	# fix the differing apostophes form textbook
	s = s.replace("’", "'")
	return s
	



