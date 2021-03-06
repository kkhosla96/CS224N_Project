from html.parser import HTMLParser

from bs4 import BeautifulSoup
import pickle
import collections

html_path = "../data/glossary_html/sadava_life_glossary.html"
outputFile = "../data/gold/sadava_life_gold.txt"
outputPickle = "../data/gold/sadava_life_gold.pkl"

with open(html_path) as f:
	soup = BeautifulSoup(f, features="html.parser")
	#class value must change for different PDFs
	#AP Biology = h4
	#testList = soup.findAll("h4")
	testList = soup.findAll("span", {"class":"cls_003"})
	# print([str(x.contents[0]).strip().lower() for x in testList])
	# testList = soup.findAll("p", {"class":"s28"})
	candidateTerms = [x.contents[0] for x in testList]
	candidateTerms = [str(x) for x in candidateTerms if x]
	candidateTerms = [x for x in candidateTerms if not x[0].isnumeric()]
	#candidateTerms = [x for x in candidateTerms if x != ")"]
	#candidateTerms = [x.strip() if x[-1] != ")" else x[:-1].strip() for x in candidateTerms]
	#candidateTerms = [x.lower() for x in candidateTerms if not x.startswith("Unit")]
	#candidateTerms = [x[:x.find(")")-1] if x.find(")") > -1 else x for x in candidateTerms]
	candidateTerms = [' '.join(x.split()) for x in candidateTerms]
	candidateTerms.sort(key=lambda x: x.lower())
	# for i in range(len(candidateTerms)):
	# 	if candidateTerms[i][0] == 'a':
	# 		break
	# candidateTerms = candidateTerms[i:]
	# lastValid = 0
	# indicesToRemove = []
	# candidateTerms = ["a", "b", "c", "b", ""]
	# for i in range(1,len(candidateTerms)):
	# 	if candidateTerms[lastValid] < candidateTerms[i]:
	# 		lastValid = i
	# 	else:
	# 		indicesToRemove.append(i)
	# print(indicesToRemove)
	print(candidateTerms[:10])
	goldTerms = collections.OrderedDict([(x,1) for x in candidateTerms])
	with open(outputFile, 'w') as f:
		for goldTerm in goldTerms:
			f.write(goldTerm + "\n")
	with open(outputPickle, 'wb') as handle:
		pickle.dump(goldTerms, handle)