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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


prank_AP_sum = 0
prank_precision_sum=0

classifier_AP_sum = 0
classifier_precision_sum=0

n_docs=0

def get_trainning_dataset(path, n_features):
    i=0
    
    
    matrix_tranning=np.array([]).reshape(0,n_features)
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.startswith('.'):

                ranking=[]
                file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
                sentences_vectors, isfs, counts_of_terms_sent, sents_without_words= ex1.sentences_ToVectorSpace(sentences, CountVectorizer())
                sentences=np.delete(sentences, sents_without_words)
                
                features_doc=calculate_features(sentences_vectors,isfs, counts_of_terms_sent)
    
                ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( training_summaries_filesPath[i])  
                    
                for sentence in sentences:
                    if sentence in ideal_summary:
                        ranking.append(1)
                    else:
                        ranking.append(0)
                                        
                features_doc[:,-1]= ranking
                matrix_tranning=np.concatenate((matrix_tranning, features_doc), axis=0)
                i+=1
             
    return matrix_tranning 




def score_real_dataset(path, trainning_dataset,  classifier):
    i=0
    
    net = classifier
    net.fit(trainning_dataset[:,:-1] , trainning_dataset[:,-1].T) #fit (X, y) samples/classes
    
    w,b=PRank_Algorithm(tranning_dataset, 5 )
           
    for root, dirs, files in os.walk(path):
        for f in files:
            #print("files", f)
            file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
            sentences_vectors, isfs, counts_of_terms_sent, sents_without_words= ex1.sentences_ToVectorSpace(sentences, CountVectorizer())
            sentences=np.delete(sentences, sents_without_words)
                
            dataset=calculate_features(sentences_vectors,isfs, counts_of_terms_sent)[:,:-1]

                        
            prank_summary,prank_summary_to_user=rank_with_Prank(dataset,  w, b, sentences)
            
            classifier_summary=predict_rank(dataset, net,sentences )

            
            ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            
            global prank_AP_sum,prank_precision_sum, classifier_AP_sum, classifier_precision_sum     
            prank_AP_sum,prank_precision_sum =  ex2.calculate_precision_recall_ap(prank_summary,ideal_summary, ideal_summary_sentences,prank_AP_sum, prank_precision_sum)
            classifier_AP_sum,classifier_precision_sum =  ex2.calculate_precision_recall_ap(classifier_summary,ideal_summary, ideal_summary_sentences,classifier_AP_sum, classifier_precision_sum)
            
            i+=1
            
    return len(files)


def calculate_features(sentences_vectors,isfs, counts_of_terms_sent):
    n_sentences=len(sentences_vectors)
    features=np.zeros((n_sentences, 5))

    #Feature graph centrality
    graph,indexes, indexes_not_linked=ex2.get_graph(sentences_vectors)
    prior=ex2.get_prior_lenSents(counts_of_terms_sent, indexes_not_linked)
    PR= calculate_improved_prank(graph, 0.15,50, prior, indexes_not_linked)
    features[:,1]=PR.ravel()
    
    #Position and Len Sentences
    features[:,0]=ex2.get_prior_PositionAndLenSents(counts_of_terms_sent, [])

    #Features TF-IDF
    doc_vector=ex1.doc_ToVectorSpace(isfs, counts_of_terms_sent)
    features[:,2]=ex2.get_prior_TFIDF(doc_vector, sentences_vectors,[])
    
    #Features position
    features[:,3]=ex2.get_prior_Position(n_sentences,[])
              
    return features


def predict_rank(dataset, net, sentences):
     predictions=net.predict_proba(dataset)     
     range = np.expand_dims(np.arange(len(dataset)), axis=1)
     concat=np.concatenate((predictions, range), axis=1)     
     sorted_concat = concat[concat[:,1].argsort()[::-1]][0:5]
     return [sentences[int(item[-1])] for item in sorted_concat]
    

def PRank_Algorithm(dataset_trainning, n_loops ):
    r = [1,0]
    n_features=dataset_trainning.shape[1]-1  
    w = np.zeros(n_features)
    #print("n_features", n_features)
    b = [0, math.inf]
    #print("len", dataset_trainning.shape)
    #print("dataset", dataset_trainning )
    
    print("len data", dataset_trainning.shape)

    for t in range(0, n_loops) :
        count_corrects=0
                
        for x in dataset_trainning:
            x=choice(dataset_trainning)   #jOEL-> FAZER RANDOM DA MELHOR RESULTADO"
            #print("x", x)
            #print("x", x[:n_features])
            predict_rank=0
            for i in range(0, len(r)):
                value = np.dot(w, x[:n_features])
                if(value < b[i]):
                    predict_rank=r[i]
                    break  
            real_rank = x[-1] # last value is the target y ([f1,f2,.., y])
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
            
    return w,b 



def rank_with_Prank(real_dataset, w, b, sentences ):
    r = [1,0]  #Important vs Non-important
    values_predicted={}
    
    for sent_index in range(len(real_dataset)) :
        value = np.dot(w, real_dataset[sent_index])
        predict_rank=0
        for i in range(0, len(r)):
            #print("b",b )
            #print("b", b[i])

            if(value < b[i]):
                #print("value", value)
                predict_rank=r[i]
                
                break
            
        if predict_rank not in values_predicted:
            values_predicted[predict_rank]={}
        values_predicted[predict_rank][sent_index]=value 
                        
    if 1 in values_predicted:
            summary, summary_to_user=show_summary(values_predicted[1], sentences, 5) #E senao houver 5?
            #if len(values_predicted[1])>=5:
                #summary, summary_to_user=show_summary(values_predicted[0], sentences, 5)
    else:
        summary, summary_to_user=show_summary(values_predicted[0], sentences, 5)
    return summary, summary_to_user

def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=False)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user  

    
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


'''  Da para remover uma vz q ja juntei tudo na mesma funcao    
def scoreWithAdaBoost(dataset, path) :
    net = RandomForestClassifier()
    
    #dataWithoutNaNs = dataset[~np.isnan(dataset).any(axis=1)]
    ncols = dataset.shape[1] # Number of features and classes
    data = dataset[0:, 0:(ncols - 1)] # Get only the features
    train = dataset[:,ncols-1].T # Get only the classes
    
        
    net.fit(data,train)
    
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
            sentences_vectors, isfs, counts_of_terms_sent, sents_without_words= ex1.sentences_ToVectorSpace(sentences, CountVectorizer())
            sentences=np.delete(sentences, sents_without_words)
                
            dataset_to_classify=calculate_features(sentences_vectors,isfs, counts_of_terms_sent)[0:, 0:(ncols - 1)]
            
            predictions=net.predict_proba(dataset_to_classify)
            range = np.expand_dims(np.arange(len(dataset_to_classify)), axis=1)
            
            
            concat=np.concatenate((predictions, range), axis=1)
            #print("predictions", concat)

            sorted_concat = concat[concat[:,1].argsort()[::-1]][0:5]
                
            summary=[sentences[int(item[-1])] for item in sorted_concat]
            
            ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            global AP_sum, precision_sum            
            AP_sum,precision_sum =  ex2.calculate_precision_recall_ap(summary,ideal_summary, ideal_summary_sentences,AP_sum, precision_sum)
            
            #break
            i+=1
            
    return len(files)
'''   

if __name__ == "__main__":
    
    training_summaries_filesPath=ex2.get_ideal_summaries_files("TeMario2006/SumariosExtractivos/.")
    
    ideal_summaries_filesPath=ex2.get_ideal_summaries_files('TeMario/Sumarios/Extratos ideais automaticos')

    tranning_dataset=get_trainning_dataset("TeMario2006/Originais/.",5)

    #n_docs=scoreWithAdaBoost( tranning_dataset,'TeMario/Textos-fonte/Textos-fonte com titulo')
    
    classifier=RandomForestClassifier()
    n_docs=score_real_dataset('TeMario/Textos-fonte/Textos-fonte com titulo', tranning_dataset, classifier)  
    
    print("\n PRANK - MAP", (prank_AP_sum / n_docs))
    print("\n PRANK - Precision", (prank_precision_sum / n_docs))
    
    print("\n choosen classifier - MAP", (classifier_AP_sum / n_docs))
    print("\n choosen classifier - Precision", (classifier_precision_sum / n_docs))
