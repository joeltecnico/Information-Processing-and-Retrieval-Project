#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:14:13 2017

@author: RitaRamos
"""

#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
import os
from os import listdir
from os.path import isfile, join
import codecs
import exercise_1
import re
import math

def counts_and_tfs_biGrams(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content):
    vec = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b')  #é dado o vocabulario dos documentos
    counts_of_terms_sent, tfs_sent=counts_and_tfs_biGrams(content, vec) #as contagens e os tfs para as frases
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_sent*isfs, isfs, vec.vocabulary_


def doc_ToVectorSpace(content, isfs,docs_vocabulary ):
    vec = CountVectorizer(vocabulary=docs_vocabulary)
    counts_of_terms_doc, tfs_doc=counts_and_tfs_biGrams([content], vec)  # as contagens e tfs para o documento
    return tfs_doc*isfs

def sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_content=[]
    docs_sentences_vectors={}
    docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content = open(os.path.join(root, f), 'rb').read().decode('iso-8859-1')
            file_with_splitLines = file_content.splitlines()
            
            sentences=[]
            for line in file_with_splitLines:
                sentences+=nltk.sent_tokenize(line)
                
            docs_sentences_vectors[i], isfs, vocabulary=sentences_ToVectorSpace(sentences)   #converter frases para vectores, usando ex 1
            docs_vectors[i]=doc_ToVectorSpace(file_content, isfs, vocabulary)                #converter doc para vector, usando ex 1 (argument2: inverse sentence frequency)
            
          
            docs_sentences[i] = sentences     #vão sendo guardadas as frases para depois calcular para ex2
            docs_content.append(file_content) #vão sendo guardado os documentos todos para depois calcular-se para ex2
            i+=1
            
            
    return docs_sentences_vectors,docs_vectors,  docs_sentences,docs_content



def calculate_cosine_for_the_ex(vector_of_docsSentences,  vectors_of_docs, number_of_docs):    
    AP_sum = 0
    precision_sum = 0
    recall_sum = 0
    cosSim_of_docs={}
    for i in range(number_of_docs):
        cosSim_of_docs[i]=exercise_1.cosine_similarity(vector_of_docsSentences[i] , vectors_of_docs[i][0])
        
        AP, precision, recall = show_summary(cosSim_of_docs[i], i, AP_sum)
        AP_sum += AP
        precision_sum += precision
        recall_sum += recall
    MAP = AP_sum / number_of_docs
    print('MAP ex1 ', MAP)
    print('Recall ex1 ', recall_sum / number_of_docs)
    print('Precision ex1 ', precision_sum / number_of_docs)

    
def show_summary(ex1_cosSim, id_doc, AP_sum):
    doc_sentences=docs_sentences[id_doc]
    ex1_summary_to_user, ex1_summary=exercise_1.show_summary(ex1_cosSim, doc_sentences)
    print('\n For Doc1: ', id_doc)
    print('\n Ex3 summary: ', ex1_summary_to_user)
    
   
    mypath = 'TeMario/Sumários/Extratos ideais automáticos'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #Isto ficaria melhor se mantivessemos uma lista com os nomes dos ficheiros...
    ##/HORRIBLE DANGER
    
    
    summary_content = open(mypath + '/' + onlyfiles[id_doc], 'rb').read().decode('iso-8859-1')
    summary_splitted = summary_content.splitlines()
    
    summary_sentences=[]
    for line in summary_splitted:
        summary_sentences+=nltk.sent_tokenize(line)
        
    RuA1 = sum(1 for x in ex1_summary_to_user if x in summary_content)
    precision1 = RuA1 / len(ex1_summary_to_user)
    recall1 = RuA1 / len(summary_sentences)
    f11 = 2 * np.float64(precision1 * recall1) / (precision1 + recall1) #Sometimes F1 can end up dividing by zero, this will give either inf or NaN as a result
    
   
    print('\nPrecision on Ex3', precision1)
    print('Recall on Ex3', recall1)
    print('F1 on Ex3', f11)
   
    
    #Calculate PA
    # Ex1
    precision_recall_curve_ex1 = []
    correct_until_now = 0
    for i in range(len(ex1_summary_to_user)) :
        if ex1_summary_to_user[i] in summary_content :
            correct_until_now +=1
            precision = correct_until_now / (i+1)
            recall = correct_until_now / len(summary_sentences)
            precision_recall_curve_ex1.append((recall, precision))
    print('Iguais ', correct_until_now)
    print(precision_recall_curve_ex1)
    
    total=0
    last_index=1
    for (R,P) in precision_recall_curve_ex1:
        part,truncated_index = math.modf(R * 10)
        truncated_index=int(truncated_index)
        for R in range(last_index, truncated_index+1):
            total += (truncated_index * 0.1) *P
            
        last_index=truncated_index+1
    
    AP = 0
    if correct_until_now > 0:
        AP = total / correct_until_now
    return AP, precision1, recall1
    


#Results:
vector_of_docsSentences,vectors_of_docs, docs_sentences, docs_content =sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
number_of_docs=len(docs_sentences)
calculate_cosine_for_the_ex(vector_of_docsSentences, vectors_of_docs,number_of_docs)


    