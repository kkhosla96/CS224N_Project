import os
import sys

SENTENCES_FOLDER = "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"
GOLD_FOLDER = "../data/gold"

















def main(sentence_file, candidates_file, gold_file):
    sentence_file_path = os.join(SENTENCES_FOLDER, sentence_file)
    sentence_file_stream = open(sentence_file_path, 'r')
    sentences = [line.strip().split() for line in sentence_file_stream if len(line.strip().split()) > 0]
    sentence_file_stream.close()
    


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python get_bert_vectors.py <sentence_file> <candidates_file> <gold_file>")
    else:
        main(*sys.argv[1:])
