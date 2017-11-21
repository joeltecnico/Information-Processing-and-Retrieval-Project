#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Joel Almeida        81609
# Matilde Gonรงalves    82091
# Rita Ramos        86274

import time
start_time = time.time()

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
import re
import operator

def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
                sentences.append(sentence)
    return file_content,sentences 


def sentences_ToVectorSpace(file_content):
    vec = CountVectorizer()
    counts_of_terms_sents=vec.fit_transform(file_content).toarray() 
    return counts_of_terms_sents

    
'''def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    return tfs_sent*isfs, isfs, counts_of_terms_sent


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs
'''

def cosine_similarity(main_sentence,sentences_vectors ):
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
    return np.array(cosSim) 

def get_graph(sentences_vectors, threshold):
    graph = defaultdict(list)
    n_sentences=len(sentences_vectors)

    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        index_of_edges=np.asarray(np.where(cos_sim>threshold))+start_index
        graph[node] += list(index_of_edges[0])
        for i in index_of_edges[0] :
            graph[i].append(node)
    return graph


def calculate_page_rank(graph, d, n_iter):
    n_docs = len(graph)
    jump_random = d / n_docs
    prob_not_dumping = 1 - d
    PR = dict.fromkeys(range(n_docs), 1/n_docs)
    for i in range(0, n_iter) :
        for node in graph :
            sum = 0
            for link in graph[node] :
                sum += PR[link]/len(graph[link])
            PR[node] = jump_random + prob_not_dumping * sum
            
        #put here the code por page ranking
    return PR
    
def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user  
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("TeMario/Textos-fonte/Textos-fonte com titulo/ce94ab10-a.txt") 
    sentences_vectors=sentences_ToVectorSpace(sentences)  
    graph=get_graph(sentences_vectors, 0.2)
    PR = calculate_page_rank(graph, 0.15, 50)
    summary, summary_to_user=show_summary(PR,sentences,5)
    print(summary_to_user)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
                                
