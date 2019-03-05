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

def calculate_precision_and_recall(predicted_file, label_file, gold_file):
	with open(predicted_file, 'rb') as f:
		predictions = pickle.load(f)
	with open(label_file, 'rb') as f:
		labels = pickle.load(f)
	with open(gold_file, 'rb') as f:
		golds = pickle.load(f)
	predicted_positive = set([' '.join(predictions[i]) for i in range(len(labels)) if labels[i] == 1])
	golds = set(golds)
	p = precision(predicted_positive, golds)
	r = recall(predicted_positive, golds)
	return (p, r)

def get_positive_predictions(predicted_file, label_file):
	with open(predicted_file, 'rb') as f:
		predictions = pickle.load(f)
	with open(label_file, 'rb') as f:
		labels = pickle.load(f)
	predicted_positive = set([' '.join(predictions[i]) for i in range(len(labels)) if labels[i] == 1])
	return predicted_positive
