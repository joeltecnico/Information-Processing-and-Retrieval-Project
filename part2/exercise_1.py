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


def sentences_ToVectorSpace(content, vec): #TF-IDF
    counts_of_terms_sent, tfs_sent, sents_without_words=counts_and_tfs(content, vec) 
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))
    return tfs_sent*isfs, isfs, counts_of_terms_sent, sents_without_words


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    sents_without_words = np.where(~counts_of_terms.any(axis=1))
    #we then remove the lines that correspond to sentences that have not words
    counts_of_terms = counts_of_terms[~np.all(counts_of_terms == 0, axis=1)]
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs, sents_without_words

def doc_ToVectorSpace(isfs, counts_of_terms_sent):
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0) 
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)
    tfs_doc=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]    
    return tfs_doc*isfs

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
        cos_sim=cosine_similarity(sentences_vectors[node],
                                  sentences_vectors[start_index:])
        index_of_edges=np.asarray(np.where(cos_sim>threshold))+start_index
        if len(index_of_edges[0])>0: #if sent is connected to others sents
            graph[node] += list(index_of_edges[0])
            for i in index_of_edges[0] :
                graph[i].append(node)
    return graph


def calculate_page_rank(graph, d, n_iter):
    n_sent = len(graph)
    if n_sent>0:
        jump_random = d / n_sent
        prob_not_dumping = 1 - d
        PR = dict.fromkeys(graph.keys(), 1/n_sent)
        for i in range(n_iter) :
            PR_new= {}
            for node in graph :
                sum_links = 0
                for link in graph[node] :
                    sum_links += PR[link]/len(graph[link])
                PR_new[node] = jump_random + (prob_not_dumping * sum_links)
            PR=PR_new
        return PR
    return {}
    
def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user  
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("datasets/TeMario/Textos-fonte/Textos-fonte com titulo/ce94ab10-a.txt") 
    sentences_vectors,isfs, counts_of_terms_sent, sents_without_words=sentences_ToVectorSpace(sentences, CountVectorizer())  
    sentences=np.delete(sentences, sents_without_words)
    
    graph=get_graph(sentences_vectors, 0.2)     
    PR = calculate_page_rank(graph, 0.15,50)
    summary, summary_to_user=show_summary(PR, sentences,5)
    print("\n Summmary", summary)
    print("\n Summmary to the user", summary_to_user)
    
    
                                
