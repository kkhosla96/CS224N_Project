import re

dataFile = "data/textbook-sentences.txt"
outputFile = "data/simple-textbook-sentences.txt"

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
	return s

out = open(outputFile, 'w')

with open(dataFile) as f:
	line = f.readline()
	while line:
		sentence = line.split("\t")[1]
		cleaned = normalize(sentence)
		out.write(cleaned + "\n")
		line = f.readline()

out.close()
