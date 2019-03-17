import os
import io
import sys
import spacy
if ".." not in sys.path:
	sys.path.append("..")
from utils import normalize

data_dir =  "../data/textbook_sentences/"
nlp = spacy.load('en')

def main():
	assert(len(sys.argv) == 3)
	num_chapters = int(sys.argv[2])
	textbook_name = sys.argv[1]
	all_sents = []
	for i in range(1,num_chapters+1):
		file = data_dir + textbook_name + "/" + textbook_name + "_chapter_" + str(i) + "_sentences.txt"
		doc_text = io.open(file, "r", encoding='utf-8').read().replace('\n', ' ')
		doc = nlp(doc_text)
		for j, sent in enumerate(doc.sents):
			#new_sent = " ".join([normalize(word.lemma_).strip() for word in sent])
			new_sent = "7." + str(i) + "." + str(j) + "\t" + str(sent)
			all_sents.append(new_sent)
		#all_sents.extend(list(doc.sents))
		print(len(list(doc.sents)))

	output_file = data_dir + textbook_name + "/" + "all_sentences_split_numbered.txt"
	with open(output_file, "w") as f:
		for sent in all_sents:
			#sent_doc = nlp(str(sent))
			#new_sent = " ".join([normalize(word.lemma_).strip() for word in sent_doc])
			f.write(sent + "\n")




if __name__ == '__main__':
	main()