import csv

data = "../data/life_glossary.csv"
outputFile = "../data/life_gold.txt"

with open(data) as data_csv:
	glossary_reader = csv.reader(data_csv)
	for row in glossary_reader:
		print(row[0].lower())

