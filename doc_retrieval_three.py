import nltk
from math import log, sqrt
import os
from collections import Counter
import time
import json
import pickle
from math import log, sqrt


##该文件会输出四个新的文件：dic_indenty.txt储存page indentifier的所有去重过后的单词，data_indenty.pickle储存page indentifer的词频
## page_number.txt储存page number，page_content.txt储存每个page的内容，内容按照.分句子后，对应每局evidence

textpath = "/wiki-pages-text/"
texttype = "txt"
textlist = os.listdir(textpath)

#根据断言的规矩
def pre_process_key(key):
    return key.split("_")


dic = {}    #每个page对应他的内容
page_numbers = {}  #每个page对应他的所有的page numbers


for text in textlist:

    print(text)
    f = open(textpath + text, "r")
    lines = f.readlines()
    for line in lines:
        z = line.split()
        key = z[0]
        number = z[1]
        if key in page_numbers.keys():
            page_numbers[key].append(number)
        else:
            l = []
            l.append(number)
            page_numbers[key] = l


        value = " ".join(z[2:])
        value = value + '\n'
        if key in dic.keys():
            dic[key] = dic[key] + value
        else:
            dic[key] = value

print("wiki文档里的标题个数为")
print(len(dic))


f= open("page_content.pkl",'wb')
pickle.dump(dic,f)

f1= open("page_number.pkl",'wb')
pickle.dump(page_numbers,f1)

raw_docs = []



keys = list(dic.keys())
for key in keys:
    raw_docs.append(pre_process_key(key))

values = list(dic.values())

vocab = {}
doc_term_freqs = []


stemmer = nltk.stem.PorterStemmer()

for raw_doc in raw_docs:

    for term in raw_doc:
        if not term in vocab.keys():
            id = len(vocab)
            vocab[term] = id
    doc_fre = Counter(raw_doc)
    doc_term_freqs.append(doc_fre)

print("Number of unique 标题 = {}".format(len(vocab)))


f2= open("dic_indenty.pkl",'wb')
pickle.dump(vocab,f2)

f3= open("data_indenty.pkl",'wb')
pickle.dump(doc_term_freqs,f3)











