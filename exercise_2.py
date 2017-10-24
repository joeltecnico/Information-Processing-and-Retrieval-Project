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

def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content, docs_vocabulary):
    vec = CountVectorizer(vocabulary=docs_vocabulary)  #é dado o vocabulario dos documentos
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #as contagens e os tfs para as frases
    idfs=np.log10(len(counts_of_terms_sent)/((counts_of_terms_sent != 0).sum(0) +1) ) # idfs com smoothing, para não dar zero!
    #print(vec.vocabulary_)
    return tfs_sent*idfs

def doc_ToVectorSpace(content, number_of_docs):
    vec = CountVectorizer()
    counts_of_terms_doc, tfs_doc=counts_and_tfs(content, vec)  # as contagens e tfs para o documento
    idfs=np.log10(number_of_docs/(counts_of_terms_doc != 0).sum(0))  # idfs= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_doc*idfs, vec.vocabulary_


def ex1_sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_content=[]
    ex1_docs_sentences_vectors={}
    ex1_docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content = open(os.path.join(root, f), 'rb').read().decode('iso-8859-1')
            file_with_splitLines = file_content.splitlines()
            
            sentences=[]
            for line in file_with_splitLines:
                sentences+=nltk.sent_tokenize(line)
            #print('Novo frases', sentences) 
            
            
            #file_content=str(file_content).replace('\n', '.')
            
            
            #file_content=re.sub(r'(\\r|\\n)+', '. ', file_content)
            
           
           
            #sentences=nltk.sent_tokenize(file_content) #o doc dividido em frases
                                         
                                         
            #print('\n\n Frases como tinhamos ', sentences)
            ex1_docs_sentences_vectors[i], isfs=exercise_1.sentences_ToVectorSpace(sentences)   #converter frases para vectores, usando ex 1
            ex1_docs_vectors[i]=exercise_1.doc_ToVectorSpace(file_content, isfs)                #converter doc para vector, usando ex 1 (argument2: inverse sentence frequency)
            
            docs_sentences[i] = sentences     #vão sendo guardadas as frases para depois calcular para ex2
            docs_content.append(file_content) #vão sendo guardado os documentos todos para depois calcular-se para ex2
            i+=1
            
    return  ex1_docs_sentences_vectors, ex1_docs_vectors, docs_sentences,docs_content


def ex2_sentences_and_docs_ToVectorSpace(docs_content,docs_sentences,number_of_docs ):
    ex2_docs_vectors, vocabulary=doc_ToVectorSpace(docs_content, number_of_docs)  #converter docs em vector usando o ex2
    ex2_docs_sentences_vectors={}                                
    for i in range(number_of_docs):
        ex2_docs_sentences_vectors[i]=sentences_ToVectorSpace(docs_sentences[i], vocabulary)  #converter frases em vector usando o ex2 
    return ex2_docs_sentences_vectors, ex2_docs_vectors
    

def calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences,  ex1_vectors_of_docs, ex2_vector_of_docsSentences, ex2_vectors_of_docs, number_of_docs  ):    
    ex1_cosSim_of_docs={}
    ex2_cosSim_of_docs={}
    for i in range(number_of_docs):
        ex1_cosSim_of_docs[i]=exercise_1.cosine_similarity(ex1_vector_of_docsSentences[i] , ex1_vectors_of_docs[i][0])
        ex2_cosSim_of_docs[i]=exercise_1.cosine_similarity(ex2_vector_of_docsSentences[i] , ex2_vectors_of_docs[i])
        show_summary_for_the_2_exs(ex1_cosSim_of_docs[i],ex2_cosSim_of_docs[i], i)

    
def show_summary_for_the_2_exs(ex1_cosSim,ex2_cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    ex1_summary_to_user, ex1_summary=exercise_1.show_summary(ex1_cosSim, doc_sentences)
    ex2_summary_to_user, ex2_summary= exercise_1.show_summary(ex2_cosSim, doc_sentences)
    print('\n For Doc1: ', id_doc)
    print('\n Ex1 summary: ', ex1_summary_to_user)
    print('\n Ex2 summary: ', ex2_summary_to_user)
    
    #print('\n ex1_summary', ex1_summary)
    
    # confusing??
    #print('\n contagem dos valores q estão correctos', sum(1 for (line, sim) in ex1_summary if line<5))  ##assumi q os resultados estão certos se forem impressas as 1º 5 linhas, visto q essas são as mais importantes na noticia  
    
    ##HORRIBE DANGER
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
    
    RuA2 = sum(1 for x in ex2_summary_to_user if x in summary_content)
    precision2 = RuA2 / len(ex2_summary_to_user)
    recall2 = RuA2 / len(summary_sentences)
    f12 = 2 * np.float64(precision2 * recall2) / (precision2 + recall2)
    
    print('\nPrecision on Ex1', precision1)
    print('Recall on Ex1', recall1)
    print('F1 on Ex1', f11)
    print('\nPrecision on Ex2', precision2)
    print('Recall on Ex2', recall2)
    print('F1 on Ex2', recall2)
    
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
    
    AP = total / correct_until_now
    


#Results:
ex1_vector_of_docsSentences, ex1_vectors_of_docs, docs_sentences, docs_content =ex1_sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
number_of_docs=len(docs_sentences)
ex2_vector_of_docsSentences, ex2_vectors_of_docs=ex2_sentences_and_docs_ToVectorSpace(docs_content, docs_sentences,number_of_docs )
calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences, ex1_vectors_of_docs,ex2_vector_of_docsSentences, ex2_vectors_of_docs,number_of_docs)

#Comentários:
    # o output do ficheiro 1 (exercise_1.py) está a ser chamado, quando devia estar a correr apenas as funcoes que chamei)
    
    
    