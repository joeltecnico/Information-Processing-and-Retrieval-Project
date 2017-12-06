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
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

prank_AP_sum = 0
prank_precision_sum=0

classifier_AP_sum = 0
classifier_precision_sum=0

n_docs=0

def get_training_dataset(path, n_features):
    i=0
    
    
    matrix_training=np.array([]).reshape(0,n_features+1)
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.startswith('.'):

                ranking=[]
                file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
                vec=CountVectorizer()
                sentences_vectors, isfs, counts_of_terms_sent, sents_without_words= ex1.sentences_ToVectorSpace(sentences, vec)
                sentences=np.delete(sentences, sents_without_words)
                
                training_dataset_doc=np.zeros((len(sentences), n_features+1))
                training_dataset_doc=calculate_features(sentences,sentences_vectors,isfs, counts_of_terms_sent, vec, training_dataset_doc)
    
                ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( training_summaries_filesPath[i])  
                    
                for sentence in sentences:
                    if sentence in ideal_summary:
                        ranking.append(1)
                    else:
                        ranking.append(0)
                                        
                training_dataset_doc[:,-1]= ranking
                matrix_training=np.concatenate((matrix_training, training_dataset_doc), axis=0)
                i+=1             
    return matrix_training 




def score_real_dataset(path, training_dataset,  classifier, n_features):
    i=0
    
    # Apply Principal Components with the same number of dimentions
    pca = PCA(n_components=n_features)
    pca.fit(training_dataset[:,:-1])
    training_pca = pca.transform(training_dataset[:,:-1])
    
    #Apply SMOTE to balance the dataset
    sm = SMOTE(kind='regular')
    X_res, y_res = sm.fit_sample(training_pca , training_dataset[:,-1].T)
    
    
    classifier.fit(X_res , y_res) #fit (X, y) samples/classes    
    
    w,b=PRank_Algorithm(training_dataset, 5 )
           
    for root, dirs, files in os.walk(path):
        for f in files:
            #print("files", f)
            file_content, sentences=ex1.getFile_and_separete_into_sentences(os.path.join(root, f))
            vec= CountVectorizer()
            sentences_vectors, isfs, counts_of_terms_sent, sents_without_words= ex1.sentences_ToVectorSpace(sentences, vec)
            sentences=np.delete(sentences, sents_without_words)
            
            dataset_doc=np.zeros((len(sentences), n_features))
            dataset_doc=calculate_features(sentences,sentences_vectors,isfs, counts_of_terms_sent, vec, dataset_doc)
            
            prank_summary,prank_summary_to_user=rank_with_Prank(dataset_doc,  w, b, sentences)

            pca.fit(dataset_doc)
            dataset_pca = pca.transform(dataset_doc)
                        
            classifier_summary=predict_rank(dataset_pca, classifier,sentences )

            
            ideal_summary,ideal_summary_sentences =ex1.getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            
            global prank_AP_sum,prank_precision_sum, classifier_AP_sum, classifier_precision_sum     
            prank_AP_sum,prank_precision_sum =  ex2.calculate_precision_recall_ap(prank_summary,ideal_summary, ideal_summary_sentences,prank_AP_sum, prank_precision_sum)
            classifier_AP_sum,classifier_precision_sum =  ex2.calculate_precision_recall_ap(classifier_summary,ideal_summary, ideal_summary_sentences,classifier_AP_sum, classifier_precision_sum)
            
            i+=1
            
    return len(files)


def calculate_features(sentences, sentences_vectors,isfs, counts_of_terms, vec, dataset):
    #n_sentences=len(sentences_vectors)
    
    #Feature graph centrality
    graph,indexes, indexes_not_linked=ex2.get_graph(sentences_vectors)
    prior=ex2.get_prior_lenSents(counts_of_terms, indexes_not_linked)
    PR= calculate_improved_prank(graph, 0.15,50, prior, indexes_not_linked)
    #dataset[:,0]=PR.ravel()
    
 
    
    #dataset[:,1]=ex2.get_prior_Position(len(sentences_vectors),[])
    #dataset[:,2]=ex2.get_prior_lenSents(counts_of_terms,[] )
    #dataset[:,3]=ex2.get_prior_PositionAndLenSents(counts_of_terms,[] )
    
    doc_vector=ex1.doc_ToVectorSpace(isfs, counts_of_terms)
    
    #dataset[:,4]=ex2.get_prior_SimilarityMostRelevantSent(doc_vector, sentences_vectors,[] ) 
    dataset[:,5]=ex2.get_prior_TFIDF(doc_vector, sentences_vectors,[])
    dataset[:,6]=ex2.get_prior_PositionAndLenSentsAndTFIDF(doc_vector, sentences_vectors,counts_of_terms,[] )




    dataset[:,7]=ex2.get_prior_termsPosition(sentences,counts_of_terms, vec,[])

    
    return dataset


def predict_rank(dataset, net, sentences):
     predictions=net.predict_proba(dataset)     
     range = np.expand_dims(np.arange(len(dataset)), axis=1)
     concat=np.concatenate((predictions, range), axis=1)     
     sorted_concat = concat[concat[:,1].argsort()[::-1]][0:5]
     return [sentences[int(item[-1])] for item in sorted_concat]
    

def PRank_Algorithm(dataset_training, n_loops ):
    r = [1,0]
    n_features=dataset_training.shape[1]-1  
    w = np.zeros(n_features)
    #print("n_features", n_features)
    b = [0, math.inf]
    #print("len", dataset_training.shape)
    #print("dataset", dataset_training )
    
    
    count_corrects=0
            
    for x in dataset_training:
        #x=choice(dataset_training)   #jOEL-> FAZER RANDOM DA MELHOR RESULTADO"
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

if __name__ == "__main__":
    
    training_summaries_filesPath=ex2.get_ideal_summaries_files("TeMario2006/SumariosExtractivos/.")
    
    ideal_summaries_filesPath=ex2.get_ideal_summaries_files('TeMario/Sumarios/Extratos ideais automaticos')
    
    n_features=8
    training_dataset=get_training_dataset("TeMario2006/Originais/.",n_features)
    
    classifier=MLPClassifier(alpha=0.01)
    n_docs=score_real_dataset('TeMario/Textos-fonte/Textos-fonte com titulo', training_dataset, classifier, n_features)  
    
    print("\n PRANK - MAP", (prank_AP_sum / n_docs))
    print("\n PRANK - Precision", (prank_precision_sum / n_docs))
    
    print("\n Multi-layer Perceptron classifier - MAP", (classifier_AP_sum / n_docs))
    print("\n Multi-layer Perceptron classifier - Precision", (classifier_precision_sum / n_docs))
    
