#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import exercise_1
import exercise_2
import re

AP_sum = 0
precision_sum = 0
recall_sum = 0

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t
    
from nltk.corpus import floresta
tsents = floresta.tagged_sents()
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
train = tsents[100:]
test = tsents[:100]
tagger0 = nltk.DefaultTagger('n')
tagger1 = nltk.UnigramTagger(train, backoff=tagger0)

def words_separation(sentences):
    words=[]
    for t in sentences:
        words.append(re.findall(r'\w+',t.strip().lower()))
    return words
   
def translate_tag(t):
    t = simplify_tag(t)
    if t == 'n' or t == 'prop':
        return 'NN'
    if t == 'adj':
        return 'JJ'
    if t == 'prp' or t == 'conj-s' or t == 'conj-c':
        return 'IN'
    else :
        return t
        
def tag_string(s) :
    sentence = ''
    for (a, b) in s:
        j = '_'.join([a, translate_tag(b)])
        sentence = ' '.join([sentence, j])
    return sentence

def counts_and_tfs(file_content, vec):
    counting = {}
    separation = words_separation(file_content)
    tagged_sentences = tagger1.tag_sents(separation)
    for i in range(len(tagged_sentences)) :
        tag_sentence = tag_string(tagged_sentences[i])
        m = re.findall(r'((( \w+_JJ)*( \w+_NN)+( \w+_IN))?( \w+_JJ)*( \w+_NN)+)+', tag_sentence, re.UNICODE)
        for group_found in m:
            group = group_found[0].strip()
            if len(group.strip().split(' ')) > 2:
                if group not in counting:
                    counting[group] = [0] * len(file_content)
                    counting[group][i] = 1
                else :
                    counting[group][i] += 1
                            
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    
    found = np.array(list(counting.values()))
    print(np.shape(found))
    print(np.shape(counts_of_terms))
    counts_of_terms = np.concatenate((counts_of_terms, found.T), axis=1)
    print(np.shape(counts_of_terms))
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def counts_and_tfs2(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs
    

def sentences_ToVectorSpace(content):
    vec = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b')  #vocabulario das frases do documento com unigrama e brigrama
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #as contagens e os tfs para as frases
    
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_sent*isfs, isfs, vec.vocabulary_


def doc_ToVectorSpace(content, isfs,docs_vocabulary ):
    vec = CountVectorizer(vocabulary=docs_vocabulary)
    counts_of_terms_doc, tfs_doc=counts_and_tfs2([content], vec)  # as contagens e tfs para o documento
    return tfs_doc*isfs

def sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_sentences_vectors={}
    docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content,sentences= exercise_1.getFile_and_separete_into_sentences(os.path.join(root, f))
           
            docs_sentences_vectors[i], isfs, vocabulary=sentences_ToVectorSpace(sentences)   #converter frases para vectores, usando ex 1
            docs_vectors[i]=doc_ToVectorSpace(file_content, isfs, vocabulary)                #converter doc para vector, usando ex 1 (argument2: inverse sentence frequency)
        
            docs_sentences[i] = sentences     #vão sendo guardadas as frases para depois calcular para ex2
            i+=1
            break
                  
    return docs_sentences_vectors,docs_vectors,  docs_sentences


def calculate_cosine_for_the_ex(vector_of_docsSentences,  vectors_of_docs, number_of_docs):    
    cosSim_of_docs={}
    for i in range(number_of_docs):
        cosSim_of_docs[i]=exercise_1.cosine_similarity(vector_of_docsSentences[i] , vectors_of_docs[i][0])
        show_summary(cosSim_of_docs[i], i)
        

def show_summary(cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    summary_to_user, summary=exercise_1.show_summary(cosSim, doc_sentences, 5)
    evaluate_summaries(summary_to_user,id_doc)


def evaluate_summaries( summary_to_user, id_doc):
    ideal_summary,ideal_summary_sentences =exercise_1.getFile_and_separete_into_sentences(ideal_summaries_filesPath[id_doc])  
    global AP_sum, precision_sum,recall_sum
    AP_sum, precision_sum,recall_sum=  exercise_2.calculate_precision_recall_ap(summary_to_user, ideal_summary, ideal_summary_sentences,AP_sum, precision_sum,recall_sum)

    
if __name__ == "__main__":
    ideal_summaries_filesPath=exercise_2.get_ideal_summaries_files('TeMario/Sumários/Extratos ideais automáticos')
    vector_of_docsSentences,vectors_of_docs, docs_sentences =sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
    number_of_docs=len(docs_sentences)
    calculate_cosine_for_the_ex(vector_of_docsSentences, vectors_of_docs,number_of_docs)
    exercise_2.print_results("exercise 3.1", (precision_sum / number_of_docs),(recall_sum / number_of_docs),(AP_sum/number_of_docs))