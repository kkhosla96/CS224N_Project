import pickle

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
	return len(predicted_terms & gold) / len(predicted_terms)

def recall(predicted_terms, gold):
	return len(predicted_terms & gold) / len(gold)

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



