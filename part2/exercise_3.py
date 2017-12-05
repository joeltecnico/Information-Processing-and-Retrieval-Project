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
import exercise_2 as ex2
import exercise_1 as ex1
import os
from random import choice
import math




def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
                sentences.append(sentence)
    return file_content,sentences 



def get_trainning_dataset(path, n_features):
    i=0
    matrix_tranning=np.array([]).reshape(0,n_features)
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.startswith('.'):
                    
                ranking=[]
                file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
                features_doc=calculate_features(sentences)
    
                #print("trainning summaries", training_summaries_filesPath[0])
                ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( training_summaries_filesPath[i])  
    
        
                for sentence in sentences:
                    if sentence in ideal_summary:
                        ranking.append(1)                        
                    else:
                        ranking.append(0)
                        
                
                features_doc[:,-1]= ranking
                            
                        
                matrix_tranning=np.concatenate((matrix_tranning, features_doc), axis=0)
    
                
    return matrix_tranning #retornas n-docs




def score_real_dataset(path, w, b):

    for root, dirs, files in os.walk(path):
        for f in files:
            #print("files", f)
            file_content, sentences=getFile_and_separete_into_sentences(os.path.join(root, f))
            dataset=calculate_features(sentences)
            rank_with_Prank(dataset,  w, b)
            
            #ideal_summary,ideal_summary_sentences =getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            #rank_with_Prank()
            #summary
            break
            
    return 0 


def calculate_features(sentences):
    n_docs=len(sentences)
    features=np.zeros((n_docs, 4))
    
    #put features
    #feature position
    #sentences_positions=ex2.get_prior_Position(n_docs,[])
    
    #features[:,0]=sentences_positions
          
    #feature centrality        
    #features[:,1]=PR.ravel()
    
    #feature tf-idf with doc
    sentences_vectors, isfs, counts_of_terms_sent= ex2.sentences_ToVectorSpace(sentences, CountVectorizer(""))
    doc_vector=ex2.doc_ToVectorSpace(isfs, counts_of_terms_sent)
    features[:,0]=ex2.get_prior_TFIDF(doc_vector, sentences_vectors,[])
    #print("isfs", isfs)
    #feature len sentences
    features[:,1]=ex2.get_prior_lenSents(counts_of_terms_sent, [])
    
    ex2_graph,ex2_priors,indexes, indexes_not_linked=ex2.priorsTFIDFS_weightsTFIDFS(sentences)

    PR= calculate_improved_prank(ex2_graph, 0.15,50,  ex2_priors, indexes_not_linked)
    
    features[:,2]=PR.ravel()

    return features


def PRank_Algorithm(dataset_trainning, n_loops ):
    r = [1,0]
    n_features=dataset_trainning.shape[1]-1  
    w = np.zeros(n_features)
    b = [0, math.inf]
    count_corrects=0
    for t in range(0, n_loops) :
        x = choice(dataset_trainning)
        predict_rank=0
        for i in range(0, len(r)):
            value = np.dot(w, x[:n_features])
            if(value < b[i]):
                predict_rank=r[i]
                break  
        real_rank = x[-1] # last value is the target y ([f1,f2,.., y])
        #print("real_rank",real_rank)
        if (predict_rank != real_rank) :
            y_r = np.zeros(len(r)-1)
            for i in range(0, len(r)-1) :
                if (real_rank <= r[i]) :
                    y_r[i] = -1
                else :
                    y_r[i] = 1
            T_r = np.zeros(len(r)-1)
            for i in range(0, len(r)-1) :
                if (  (np.dot(w, x[:n_features]) -b[i] ) * y_r[i] <= 0) :  
                    T_r[i] = y_r[i]
                else :
                    T_r[i] = 0
    
            w = w + (np.sum(T_r))*(x[:n_features])
    
            for i in range(0, len(r)-1) :
                b[i] = b[i] - T_r[i]
        else:
            count_corrects+=1
    print("count_corrects", count_corrects)  
    return w,b 



def rank_with_Prank(real_dataset, w, b ):
    r = [1,0]  #Important vs Non-important
    n_features=real_dataset.shape[1]-1
    values_predicted={}
    

    for x in real_dataset :
        value = np.dot(w, x[:n_features])
        predict_rank=0
        for i in range(0, len(r)):
            #print("b",b )
            #print("b", b[i])

            if(value < b[i]):
                #print("value", value)
                predict_rank=r[i]
                
                break
            
        if predict_rank not in values_predicted:
            values_predicted[predict_rank]=[value]
        else:
            values_predicted[predict_rank].append(value)
   
    #print("results", values_predicted )
    
    

    
    resuls=(sorted(values_predicted[1]))
    
    
    #resuls+=(sorted(values_predicted[0]))
    #print("results", values_predicted )

    return resuls[0:5]



    
def calculate_improved_prank(graph, damping, n_iter, priors, indexes_not_linked):
    transition_probs=graph/np.sum(graph,axis=0) 
    n_docs=len(transition_probs)
    #Compute Matrix -> 1-d[Transition_Probabilities] + d* [priors] 
    matrix=  (((1-damping)*transition_probs).T + (damping)*(priors)).T #(since prior is a scaler, sum it up to each collum of Transition probs)
    r=np.ones((n_docs, 1))/n_docs    
    #print("\n Matrix", matrix)
    for i in range(n_iter) :
        r_new=matrix.dot(r)
        r=r_new
        
    for i in indexes_not_linked[0]:
        r= np.insert(r, i, 0 , axis=0)
    return r
    
'''
def get_pr_for_allSentences(PR, sentences_not_linked):
    PR=(list(PR.values()))
    
    for i in sentences_not_linked[0]:
        PR= np.insert(PR, i, 0 , axis=0)
    return PR
'''




'''
def priorsTFIDFS(sentences):
    sentences_vectors, isfs, counts_of_terms_sent= sentences_ToVectorSpace(sentences, CountVectorizer())
    doc_vector=doc_ToVectorSpace(isfs, counts_of_terms_sent)
    priors_cos=ex1.cosine_similarity(doc_vector[0], sentences_vectors)
    priors_cos=(np.expand_dims(priors_cos, axis=0)
    return non_uniform_weights/np.sum(non_uniform_weights)
'''


if __name__ == "__main__":
    #file_content, sentences=getFile_and_separete_into_sentences("script1.txt")
    #print(file_content)
    
    #sentences_vectors, isfs, counts_of_terms_sent=exercise_1_matrix.sentences_ToVectorSpace(sentences)  
    #sentences_vectors=exercise_1_matrix.sentences_ToVectorSpace(sentences)  
    #graph=exercise_1_matrix.get_graph(sentences_vectors, 0.2)     
    #print("\n Graph\n", graph)
    training_summaries_filesPath=ex2.get_ideal_summaries_files("TeMario2006/SumariosExtrativos/.")
    
    tranning_dataset=get_trainning_dataset("TeMario2006/Originais/.",4)
    w,b=PRank_Algorithm(tranning_dataset, 40000 )
    print("w", w)
    score_real_dataset('TeMario/Textos-fonte/Textos-fonte com titulo', w,b)  
    


    #calculate_features()
    '''
    features=calculate_features(sentences)
    print("FEATURES",features )
    '''
