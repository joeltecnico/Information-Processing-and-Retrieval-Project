#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Joel Almeida        81609
# Matilde Gonรงalves    82091
# Rita Ramos        86274

import time
start_time = time.time()

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import operator
import os
import exercise_1

ex1_AP_sum = 0

ex2_AP_sum = 0
ex1_precision_sum=0
ex2_precision_sum=0


stop_words=nltk.corpus.stopwords.words('portuguese')

'''
def remove_stop_words(sentences): #para cada frase, vamos retirar as stop words
    words=[]
    for sentence in sentences:
        words_of_sentence=[word for word in re.findall(r'\w+', sentence.lower()) if word
            not in arrayStopWords]
        if len(words_of_sentence) >1: #Apos remover stopWrds, verificar se a frase não ficou vazia
            words.append(words_of_sentence)
    return words

'''

def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences_without_stop_words=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            words_without_stop_words=[word for word in re.findall(r'\w+', sentence.lower()) if word
            not in stop_words]
            if len(words_without_stop_words)>0: #check:not only pontuaction
                sentence=' '.join(words_without_stop_words)
                sentences_without_stop_words.append(sentence)
    file_without_stop_words=' '.join(sentences_without_stop_words)
    
    #print ("\n File content", file_content)
    return file_without_stop_words,sentences_without_stop_words


def get_ideal_summaries_files(path):
    ideal_summaries_filesPath={}
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            ideal_summaries_filesPath[i]=os.path.join(root, f)
            i+=1
        return ideal_summaries_filesPath

def read_docs(path):
    i=0
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content, sentences=getFile_and_separete_into_sentences(os.path.join(root, f))
    
            ex1_sents_represented=exercise_1.sentences_ToVectorSpace(sentences)  
            ex1_graph=exercise_1.get_graph(ex1_sents_represented, 0.2)     
            ex1_PR = exercise_1.calculate_page_rank(ex1_graph, 0.15, 50)
            ex1_summary, ex1_summary_to_user=exercise_1.show_summary(ex1_PR,sentences,5)
            
            
            #ex2_summary, ex2_summary_to_user=PR_priorsPosition_weightsTFIDFS(sentences)
            #ex2_summary, ex2_summary_to_user=PR_priorsTFIDFS_weightsTFIDFS(sentences)
            #ex2_summary, ex2_summary_to_user=PR_priorsLenSents_weightsTFIDFS(sentences)
            ex2_summary, ex2_summary_to_user=PR_priorsPositionAndLenSents_weightsTFIDFS(sentences)
            #ex2_summary, ex2_summary_to_user=PR_priorsPosition_weightsBM25(sentences)


            print("\nDoc ",i, ": \n\nEx1- Summary to user:", ex1_summary_to_user, 
              ": \n\nEx2- Summary to user:", ex2_summary_to_user)
            
            ideal_summary,ideal_summary_sentences =getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            
            print("ideal_summary", ideal_summary)
            
            global ex1_AP_sum,  ex2_AP_sum, ex1_precision_sum,ex2_precision_sum
            if(len(ex1_summary))>0:
                ex1_AP_sum,ex1_precision_sum =  calculate_precision_recall_ap(ex1_summary,ideal_summary, ideal_summary_sentences,ex1_AP_sum, ex1_precision_sum)
            ex2_AP_sum, ex2_precision_sum=  calculate_precision_recall_ap(ex2_summary,ideal_summary, ideal_summary_sentences,ex2_AP_sum, ex2_precision_sum)
            
            #break
            i+=1
    
    return len(files)  #retornas n-docs
    

def PR_priorsPosition_weightsTFIDFS(sentences):
    sentences_vectors, isfs, counts_of_terms_sent= sentences_ToVectorSpace(sentences)
    graph,indexes, indexes_sents_not_linked=get_graph(sentences_vectors)
    priors=get_prior_Position(len(sentences_vectors),indexes_sents_not_linked)
    PR=calculate_improved_prank(graph, 0.15,50,  priors, indexes)
    print("\n SOMA", sum(list(PR.values())))
    summary, summary_to_user=exercise_1.show_summary(PR,sentences,5)
    return summary, summary_to_user
    
def PR_priorsTFIDFS_weightsTFIDFS(sentences):
    sentences_vectors, isfs, counts_of_terms_sent= sentences_ToVectorSpace(sentences)
    graph,indexes, indexes_sents_not_linked=get_graph(sentences_vectors)
    doc_vector=doc_ToVectorSpace(isfs, counts_of_terms_sent)
    priors=get_prior_TFIDF(doc_vector, sentences_vectors, indexes_sents_not_linked )    
    PR=calculate_improved_prank(graph, 0.15,50,  priors, indexes)
    print("\n SOMA", sum(list(PR.values())))
    summary, summary_to_user=exercise_1.show_summary(PR,sentences,5)
    return summary, summary_to_user

def PR_priorsLenSents_weightsTFIDFS(sentences):
    sentences_vectors, isfs, counts_of_terms_sent= sentences_ToVectorSpace(sentences)
    graph,indexes, indexes_sents_not_linked=get_graph(sentences_vectors)
    priors=get_prior_lenSents(counts_of_terms_sent, indexes_sents_not_linked)
    PR=calculate_improved_prank(graph, 0.15,50,  priors, indexes)
    print("\n SOMA", sum(list(PR.values())))
    summary, summary_to_user=exercise_1.show_summary(PR,sentences,5)
    return summary, summary_to_user

def PR_priorsPositionAndLenSents_weightsTFIDFS(sentences):
    sentences_vectors, isfs, counts_of_terms_sent= sentences_ToVectorSpace(sentences)
    graph,indexes, indexes_sents_not_linked=get_graph(sentences_vectors)
    priors=get_prior_PositionAndLenSents(counts_of_terms_sent, indexes_sents_not_linked)
    PR=calculate_improved_prank(graph, 0.15,50,  priors, indexes)
    print("\n SOMA", sum(list(PR.values())))
    summary, summary_to_user=exercise_1.show_summary(PR,sentences,5)
    return summary, summary_to_user



def get_prior_TFIDF(doc_vector, sentences_vectors,indexes_not_linked ):
    priors_cos=cosine_similarity(doc_vector[0], sentences_vectors)
    priors_cos=np.expand_dims(priors_cos, axis=0)
    priors_cos=np.delete(priors_cos, indexes_not_linked,1) #delete rows with just zeros
    return priors_cos/np.sum(priors_cos)

def get_prior_Position(n_docs,indexes_not_linked):
    priors_pos=np.arange(n_docs, 0, -1)
    priors_pos=np.expand_dims(priors_pos, axis=0)
    priors_pos=np.delete(priors_pos, indexes_not_linked,1) 
    return priors_pos/np.sum(priors_pos)

def get_prior_lenSents(counts_of_terms_sent,indexes_not_linked ): # dar + importancia as frases q tem + termos
    priors_len=(counts_of_terms_sent != 0).sum(1)    
    priors_len=np.expand_dims(priors_len, axis=0)
    priors_len=np.delete(priors_len, indexes_not_linked,1) 
    return priors_len/np.sum(priors_len)

def get_prior_PositionAndLenSents(counts_of_terms_sent,indexes_not_linked ): # dar + importancia as frases q tem + termos
    priors_len=np.arange(len(counts_of_terms_sent), 0, -1)* (counts_of_terms_sent != 0).sum(1) 
    print("priors_len", priors_len)
    priors_len=np.expand_dims(priors_len, axis=0)
    priors_len=np.delete(priors_len, indexes_not_linked,1) 
    return priors_len /np.sum(priors_len)
    #return priors_len/np.sum(priors_len)





def PR_priorsPosition_weightsBM25(sentences):
    sentences_vectors,counts_of_terms= get_score_BM5(sentences)
    graph,indexes, indexes_sents_not_linked=get_graph(sentences_vectors)
    priors=get_prior_Position(len(sentences_vectors),indexes_sents_not_linked)
    PR=calculate_improved_prank(graph, 0.15,50,  priors, indexes)
    print("\n SOMA", sum(list(PR.values())))
    summary, summary_to_user=exercise_1.show_summary(PR,sentences,5)
    return summary, summary_to_user

def get_score_BM5(content):
    k=1.2
    b=0.75
    
    vec = CountVectorizer()
    counts_of_terms=vec.fit_transform(content).toarray()
    
    nominator=counts_of_terms*(k+1)
    length_of_sentences_D=counts_of_terms.sum(1)
    number_sentences=len(counts_of_terms)
    avgdl= sum(length_of_sentences_D)/number_sentences
    denominator=counts_of_terms+(k*(1-b+b*((length_of_sentences_D)/(avgdl))))[:, None] 
    score_BM5_without_ISF=nominator/denominator
    
    N=len(counts_of_terms) #numero de frases
    nt=(counts_of_terms!=0).sum(0)  #nº vezes q o termo aparece nas frases
    isfs=np.log10((N-nt+0.5)/(nt+0.5))  # inverve sentence frequency   
         
    return score_BM5_without_ISF*isfs, counts_of_terms



def get_graph(sentences_vectors):
    n_sentences=len(sentences_vectors)
    tri_matrix=np.triu(np.zeros((n_sentences,n_sentences)),-1)
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        tri_matrix[node,start_index:]=cos_sim
    #print( "GRAP MEMSO \n ", tri_matrix+tri_matrix.T)
    
    graph=tri_matrix+tri_matrix.T
    #print("Graph\n ", graph)
    #print("graph before", len(graph))
    indexes_sents_not_linked=np.where(~graph.any(axis=0)) #collum with just zeros
    #print("senteces not linked", indexes_sents_not_linked[0])
    indexes_sents=np.delete(np.arange(len(graph)),indexes_sents_not_linked)
    #print("indexes", indexes_sents)
    graph=np.delete(graph, indexes_sents_not_linked,0) #delete rows with just zeros
    graph=np.delete(graph, indexes_sents_not_linked,1) #delete collumns with just zeros
    #print("graph after", len(graph))
    #print("Graph\n", graph)

    return graph,indexes_sents, indexes_sents_not_linked


    
def calculate_improved_prank(graph, damping, n_iter, priors, indexes):
    transition_probs=graph/np.sum(graph,axis=0) 
    n_docs=len(transition_probs)
    #Compute Matrix -> 1-d[Transition_Probabilities] + d* [priors] 
    matrix=  (((1-damping)*transition_probs).T + (damping)*(priors)).T #(since prior is a scaler, sum it up to each collum of Transition probs)
    r=np.ones((n_docs, 1))/n_docs    
    print("\n Matrix", matrix)
    for i in range(n_iter) :
        r_new=matrix.dot(r)
        r=r_new
    return dict(zip(indexes, r))


def calculate_precision_recall_ap(summary, ideal_summary_allContent,ideal_summary_sentences,
            AP_sum, precision_sum ):
    
    R=len(ideal_summary_sentences)  #documento relevante
    
    RuA = sum(1 for x in summary if x in ideal_summary_allContent)
    precision= RuA / len(summary)
    print("Precision", precision)
    precision_sum+=precision

    correct_until_now = 0
    precisions = 0
    for i in range(len(summary)) :
        if summary[i] in ideal_summary_allContent :
            correct_until_now +=1
            precisions+= correct_until_now / (i+1)
                    
    AP_sum+=(precisions/R)
    return AP_sum, precision_sum


def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    #return tfs_sent*isfs, isfs, counts_of_terms_sent
    return tfs_sent*isfs, isfs, counts_of_terms_sent


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs

def doc_ToVectorSpace(isfs, counts_of_terms_sent):
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0) #summing the terms counts of each sentence
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)  #(lines=documents, cols=terms) 
    tfs_doc=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return tfs_doc*isfs

def cosine_similarity(main_sentence,sentences_vectors ):
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
    return np.array(cosSim) 





def readTest():
    file_content,sentences=getFile_and_separete_into_sentences("script1.txt")
    
    ex1_sents_represented=exercise_1.sentences_ToVectorSpace(sentences)  
    ex1_graph=exercise_1.get_graph(ex1_sents_represented, 0.2)     
    ex1_PR = exercise_1.calculate_page_rank(ex1_graph, 0.15, 50)
    ex1_summary, ex1_summary_to_user=exercise_1.show_summary(ex1_PR,sentences,5)
    
   
    ex2_summary, ex2_summary_to_user=PR_priorsTFIDFS_weightsTFIDFS(sentences)
    
    print("\nDoc ",0, ": \n\nEx1- Summary to user:", ex1_summary_to_user, 
      ": \n\nEx2- Summary to user:", ex2_summary_to_user)

    return 0


        
        
if __name__ == "__main__":
    
    ideal_summaries_filesPath=get_ideal_summaries_files('TeMario/Sumarios/Extratos ideais automaticos')
    n_docs=read_docs('TeMario/Textos-fonte/Textos-fonte com titulo')       
    print("\n exercise 1- MAP", (ex1_AP_sum / n_docs))
    print("\n exercise 1- Precision", (ex1_precision_sum / n_docs))

    print("\n exercise 2- MAP", (ex2_AP_sum / n_docs))
    print("\n exercise 2- Precision", (ex2_precision_sum / n_docs))

    
    #readTest()
