#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import exercise_1
import math

ex1_AP_sum = 0
ex1_precision_sum = 0
ex1_recall_sum = 0
#ex1_f1=0
ex2_AP_sum = 0
ex2_precision_sum = 0
ex2_recall_sum = 0
#ex2_f1=0


def get_file_separete_into_sentences(f):
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        sentences+=nltk.sent_tokenize(line)
    return sentences, file_content 

def get_ideal_summaries_files(path):
    ideal_summaries_filesPath={}
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            ideal_summaries_filesPath[i]=os.path.join(root, f)
            i+=1
        return ideal_summaries_filesPath

def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content, docs_vocabulary):
    vec = CountVectorizer(vocabulary=docs_vocabulary)  #é dado o vocabulario dos documentos
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #as contagens e os tfs para as frases
    idfs=np.log10(len(counts_of_terms_sent)/((counts_of_terms_sent != 0).sum(0) +1) ) # idfs com smoothing, para não dar zero!
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
            sentences,file_content= get_file_separete_into_sentences(os.path.join(root, f))
           
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
    

def calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences,  ex1_vectors_of_docs, ex2_vector_of_docsSentences, ex2_vectors_of_docs, number_of_docs):    
    ex1_cosSim_of_docs={}
    ex2_cosSim_of_docs={}
    for i in range(number_of_docs):
        ex1_cosSim_of_docs[i]=exercise_1.cosine_similarity(ex1_vector_of_docsSentences[i] , ex1_vectors_of_docs[i][0])
        ex2_cosSim_of_docs[i]=exercise_1.cosine_similarity(ex2_vector_of_docsSentences[i] , ex2_vectors_of_docs[i])
        show_summary_for_the_2_exs(ex1_cosSim_of_docs[i],ex2_cosSim_of_docs[i], i)

    
def show_summary_for_the_2_exs(ex1_cosSim,ex2_cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    ex1_summary_to_user, ex1_summary=exercise_1.show_summary(ex1_cosSim, doc_sentences, 5)
    ex2_summary_to_user, ex2_summary= exercise_1.show_summary(ex2_cosSim, doc_sentences, 5)
    print('\n For Doc1: ', id_doc)
    print('\n Ex1 summary: ', ex1_summary_to_user) 
    summary_sentences,summary_content =get_file_separete_into_sentences(ideal_summaries_filesPath[id_doc])  
    global ex1_AP_sum, ex1_precision_sum,ex1_recall_sum, ex2_AP_sum, ex2_precision_sum,ex2_recall_sum
    ex1_AP_sum, ex1_precision_sum,ex1_recall_sum=  calculate_precision_recall_f1_ap(ex1_summary_to_user, summary_content, summary_sentences,ex1_AP_sum, ex1_precision_sum,ex1_recall_sum)
    print('\n Ex2 summary: ', ex2_summary_to_user)
    ex2_AP_sum, ex2_precision_sum,ex2_recall_sum= calculate_precision_recall_f1_ap(ex2_summary_to_user, summary_content, summary_sentences,ex2_AP_sum, ex2_precision_sum,ex2_recall_sum)
    

def calculate_precision_recall_f1_ap(summary_to_user, ideal_summary,summary_sentences,AP_sum ,precision_sum,recall_sum ):
    RuA = sum(1 for x in summary_to_user if x in ideal_summary)
    recall = RuA / len(summary_sentences)
    precision = RuA / len(summary_to_user)
    
    print("\n Precision", precision)
    print("\n Recall", recall)
    recall_sum+=recall
    precision_sum+= precision
                  
    precision_recall_curve = []
    correct_until_now = 0
    for i in range(len(summary_to_user)) :
        if summary_to_user[i] in ideal_summary :
            correct_until_now +=1
            precision_curve= correct_until_now / (i+1)
            recall_curve = correct_until_now / len(summary_sentences)
            precision_recall_curve.append((recall_curve, precision_curve))
    print(precision_recall_curve)
    
    total=0
    last_index=1
    for (R,P) in precision_recall_curve:
        part,truncated_index = math.modf(R * 10)
        truncated_index=int(truncated_index)
        for R in range(last_index, truncated_index+1):
            total += (truncated_index * 0.1) *P
            
        last_index=truncated_index+1   
    AP = 0
    if correct_until_now > 0:
        AP = total / correct_until_now               
    AP_sum+=AP              
    
    return AP_sum ,precision_sum,recall_sum
#, recall, f, AP


def print_results():
    ex1_precision_mean=(ex1_precision_sum / number_of_docs)
    ex2_precision_mean=(ex2_precision_sum / number_of_docs)
    ex1_recall_mean=(ex1_recall_sum / number_of_docs)
    ex2_recall_mean=(ex2_recall_sum / number_of_docs)

    print("\n Results of exercise 1: \n Precision: ", (ex1_precision_sum / number_of_docs), "\n Recall:",  (ex1_recall_sum / number_of_docs), "\n F1:",  (2 * (ex1_precision_mean * ex1_recall_mean) / (ex1_precision_mean + ex1_recall_mean)), '\n MAP: ', (ex1_AP_sum / number_of_docs))
    print("\n Results of exercise 2: \n Precision: ", (ex2_precision_sum / number_of_docs), "\n Recall:",  (ex2_recall_sum / number_of_docs), "\n F1:",  ( 2 * (ex2_precision_mean * ex2_recall_mean) / (ex2_precision_mean + ex2_recall_mean)), '\n MAP: ', (ex2_AP_sum / number_of_docs))
  


#Results:
ideal_summaries_filesPath=get_ideal_summaries_files('TeMario/Sumários/Extratos ideais automáticos')
ex1_vector_of_docsSentences, ex1_vectors_of_docs, docs_sentences, docs_content =ex1_sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
number_of_docs=len(docs_sentences)
ex2_vector_of_docsSentences, ex2_vectors_of_docs=ex2_sentences_and_docs_ToVectorSpace(docs_content, docs_sentences,number_of_docs )
calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences, ex1_vectors_of_docs,ex2_vector_of_docsSentences, ex2_vectors_of_docs,number_of_docs)
print_results()
#Comentários:
    # o output do ficheiro 1 (exercise_1.py) está a ser chamado, quando devia estar a correr apenas as funcoes que chamei)
    
    
    