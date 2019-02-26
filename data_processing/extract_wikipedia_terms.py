from bs4 import BeautifulSoup
import requests

outFile = "../data/wikipedia_terms.txt"

URL = "https://en.wikipedia.org/wiki/Glossary_of_biology"
html = requests.get(URL).text
PARSED_HTML = BeautifulSoup(html)

glossary_found = PARSED_HTML.body.find_all('dt', attrs={'class': 'glossary'})

with open(outFile, 'w') as f:
    for found in glossary_found:
        f.write(found.text + "\n")
    # print(found.text)
