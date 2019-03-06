from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

dataFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences_simple.txt"
outputFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences_simple_lemmatized.txt"

def lemmatize_sentence(sent, l):
	return ' '.join([l.lemmatize(word) for word in sent.split(" ")])

with open(dataFile, "r") as in_file:
	with open(outputFile, "w") as out_file:
		for line in in_file:
			lemma_line = lemmatize_sentence(line.strip(), lemma)
			out_file.write(lemma_line + "\n")
