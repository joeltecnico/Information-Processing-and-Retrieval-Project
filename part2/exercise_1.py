#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Joel Almeida        81609
# Matilde Gonรงalves    82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

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


def cosine_similarity(main_sentence,sentences_vectors ):
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
    return np.array(cosSim) 

def get_graph(sentences_vectors, threshold):
    n_sentences=len(sentences_vectors)
    tri_matrix=np.triu(np.zeros((n_sentences,n_sentences)),-1)
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        index_of_edges=np.asarray(np.where(cos_sim>threshold))+start_index
        tri_matrix[node,index_of_edges]=1
    return tri_matrix+tri_matrix.T
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt") 
    sentences_vectors=sentences_ToVectorSpace(sentences)  
    graph=get_graph(sentences_vectors, 0.2)     
    print(graph)      
                                
