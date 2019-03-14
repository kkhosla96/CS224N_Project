import spacy
import sys
if ".." not in sys.path:
	sys.path.append('..')
from utils import normalize

nlp = spacy.load('en_core_web_sm')

#dataFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences_simple.txt"
#outputFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences_simple_lemmatized.txt"

dataFile = "../data/gold/openstax_biology/openstax_biology_chapters4to11_gold.txt"
outputFile = "../data/gold/openstax_biology/openstax_biology_chapters4to11_gold_final.txt"

def lemmatize_sentence(sent, nlp):
	tags = nlp(sent)
	return ' '.join([normalize(word.lemma_).strip() for word in tags])

def lemmatize_file(infile, outfile, nlp):
	with open(infile, "r") as in_file:
		with open(outfile, "w") as out_file:
			for line in in_file:
				lemma_line = lemmatize_sentence(line.strip(), nlp)
				out_file.write(lemma_line + "\n")


lemmatize_file(dataFile, outputFile, nlp)
