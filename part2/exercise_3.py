#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:21:03 2017

@author: RitaRamos
"""

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import operator
import exercise_1_matrix




def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
                sentences.append(sentence)
    return file_content,sentences 

def calculate_features(sentences, n_features):
    n_docs=len(sentences)
    features=np.zeros((n_docs, n_features))
    sentences_positions=np.arange(1,n_docs+1)
    features[:,0]=sentences_positions
            
    features[:,1]=PR.ravel()
    return features
    
def calculate_page_rank(graph, damping, n_iter):
    n_docs_before=len(graph)
    sentences_not_linked=np.where(~graph.any(axis=0))
    print("senteces not linked", sentences_not_linked[0])
    indexes= np.arange(len(graph))
    indexes=np.delete(indexes,sentences_not_linked)
    print("indexes", indexes)
    graph=np.delete(graph, sentences_not_linked,0) 
    graph=np.delete(graph, sentences_not_linked,1)
    print("Graph agora sim", graph)
    
    M=graph/np.sum(graph,axis=0)
    n_docs=len(M)
    N=np.full((n_docs, n_docs), n_docs)
    matrix=(1-damping)*M+(damping)*(1/N)
    r=np.ones((n_docs, 1))/n_docs
    #err=1
    for i in range(n_iter) :
        r_new=matrix.dot(r)
        r=r_new
    
    print("R", r)
    zeros=np.zeros((len(sentences_not_linked[0]),1))
    print("zeros", zeros)
    print("sentences not linkes", sentences_not_linked)
    for i in sentences_not_linked[0]:
        r= np.insert(r, i, 0 , axis=0)
    print("r", r)
    return r
    
'''
def get_pr_for_allSentences(PR, sentences_not_linked):
    PR=(list(PR.values()))
    
    for i in sentences_not_linked[0]:
        PR= np.insert(PR, i, 0 , axis=0)
    return PR
'''

if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt")
    print(file_content)
    #sentences_vectors, isfs, counts_of_terms_sent=exercise_1_matrix.sentences_ToVectorSpace(sentences)  
    sentences_vectors=exercise_1_matrix.sentences_ToVectorSpace(sentences)  
    graph=exercise_1_matrix.get_graph(sentences_vectors, 0.2)     
    print("\n Graph\n", graph)
    
    PR= calculate_page_rank(graph, 0.15, 50)

    features=calculate_features(sentences,2)
    print("FEATURES",features )
    
