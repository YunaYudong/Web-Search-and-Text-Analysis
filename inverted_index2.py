

import pickle
import json
from collections import Counter
from math import log, sqrt
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import numpy

import string
import re
from allennlp import pretrained

predictor = pretrained.decomposable_attention_with_elmo_parikh_2017()

with open("data_indenty.pkl",'rb') as f:
    doc_term_freqs = pickle.load(f)

with open("dic_indenty.pkl",'rb') as f:
    vocab = pickle.load(f)

with open("page_content.pkl",'rb') as f3:
    page_content = pickle.load(f3)

with open("page_number.pkl",'rb') as f4:
    page_number = pickle.load(f4)





vocab_words = list(vocab.keys())

print(len(doc_term_freqs))
print(len(vocab))

print(vocab["Company"])


class InvertedIndex:
    def __init__(self, vocab, doc_term_freqs):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0

        for docid, term_freqs in enumerate(doc_term_freqs):
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1

            for term, freq in term_freqs.items():
                term_id = vocab[term]
                self.doc_ids[term_id].append(docid)
                self.doc_term_freqs[term_id].append(freq)
                self.doc_freqs[term_id] += 1

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        return self.doc_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.doc_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]





def query_tfidf(query, index, k=1):

    scores = Counter()

    for term in query:


        doc_id_list = index.docids(term)


        doc_term_freq_list = index.freqs(term)

        idf = log(index.num_docs() / index.f_t(term))
        for i in range(len(doc_id_list)):
            term_freq = doc_term_freq_list[i]
            doc_id = doc_id_list[i]
            tf = log(1 + term_freq)
            length = index.doc_len[doc_id]
            score = tf * idf / sqrt(length)
            scores[doc_id] = scores[doc_id] + score


    return scores.most_common(k)

## 提取claim的关键词
def preprocessing_claim(claim,words):
    result = []
    claim_words = nltk.word_tokenize(claim)
    for word in claim_words:
        if not word.islower() and word.isalpha() and word in words  :
            result.append(word)
    return result


 # sentence selection的打分模型
def sentence_claim(sentence,claim):
    sentence_word = nltk.word_tokenize(sentence)
    claim_word = nltk.word_tokenize(claim)
    num = len(set(sentence_word).intersection(set(claim_word)))



def analyze_entailmemt(entaiment):
    num = entaiment.index(max(entaiment))
    if num == 0:
        return "SUPPORTS"
    if num == 1:
        return "REFUTES"
    if num == 2:
        return "NOT ENOUGH INFO"
    return "啊啊啊啊啊，没有！！"



def analyze_evidence(sentence,claim):

    sentence_word = nltk.word_tokenize(sentence)

    claim_words = nltk.word_tokenize(claim)

    return len(set(sentence_word).intersection(set(claim_words)))






# 简历倒排索引表

invindex = InvertedIndex(vocab, doc_term_freqs)
print("建立倒排表完成....")

with open("devset.json",'r') as load_f:
    docsss = json.load(load_f)

out_put = {}


for id ,value in docsss.items():
    content = {}

    claim = value["claim"]

    content["claim"] = claim



    query = preprocessing_claim(claim, vocab_words)

    results = query_tfidf(query, invindex)

    print(results)


    page_text = []
    page_numbers = []
    page_indentifier = []

    for item in results:
        page_id = item[0]
        values = list(page_content.values())
        keys = list(page_content.keys())

        page_number_value = (list(page_number.values())[page_id])
        page_numbers.append(page_number_value)

        text = values[page_id]
        page_text.append(text)
        identifier = keys[page_id]
        page_indentifier.append(identifier)

    splited_text = []
    for text in page_text:
        text = text.split('\n')
        new_text = []
        for txt in text:
            if txt != "":
                new_text.append(txt)

        splited_text.append(new_text)

    print(page_numbers)
    print(page_indentifier)
    print(splited_text)

    for text in splited_text:
        similarity = []
        for i in range(len(text)):
            num = analyze_evidence(text[i], claim)
            similarity.append(num)

    max_index = similarity.index(max(similarity))
    if splited_text == []:

        content["label"] = "NOT ENOUGH INFO"
        content["evidence"] = []
        out_put[id] = content

    else:
        selected_sentence = splited_text[0][max_index]

        result = predictor.predict_json({"premise": selected_sentence, "hypothesis": claim})

        predict_result = result["label_probs"]

        final_evidence = []
        final_label = analyze_entailmemt(predict_result)

        if final_label == "SUPPORTS" or "REFUTES":
            evidence = []
            evidence.append(page_indentifier[0])
            evidence.append(int(page_numbers[0][max_index]))
            final_evidence.append(evidence)
        if final_label == "NOT ENOUGH INFO":
            final_evidence = []

        content["label"] = final_label
        content["evidence"] = final_evidence
        out_put[id] = content



json_str = json.dumps(out_put,indent=1)
with open('outputt.json', 'w') as json_file:
    json_file.write(json_str)























































