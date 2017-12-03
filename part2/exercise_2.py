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


def get_ideal_summaries_files(path):
    ideal_summaries_filesPath={}
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            ideal_summaries_filesPath[i]=os.path.join(root, f)
            i+=1
        return ideal_summaries_filesPath


def get_ideal_summaries_files(path):
    ideal_summaries_filesPath={}
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            ideal_summaries_filesPath[i]=os.path.join(root, f)
            i+=1
        return ideal_summaries_filesPath

def read_docs(path, function_represent_sents, function_graph, function_prior):
    i=0
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content,sentences=exercise_1.getFile_and_separete_into_sentences(os.path.join(root, f))
    
            ex1_sents_represented=sentences_ToVectorSpace(sentences)  
            ex1_graph=exercise_1.get_graph(ex1_sents_represented, 0.2)     
            ex1_PR = exercise_1.calculate_page_rank(ex1_graph, 0.15, 50)
            ex1_summary, ex1_summary_to_user=exercise_1.show_summary(ex1_PR,sentences,5)
            
            ex2_sents_represented=function_represent_sents(sentences)
            ex2_graph,indexes, indexes_sents_not_linked=function_graph(ex2_sents_represented)
            priors=function_prior(len(ex2_graph))
            ex2_PR=calculate_improved_prank(ex2_graph, 0.15,50,  priors, indexes)
            ex2_summary, ex2_summary_to_user=exercise_1.show_summary(ex2_PR,sentences,5)


            print("DOC:", i, "\n")
            
            ideal_summary,ideal_summary_sentences =exercise_1.getFile_and_separete_into_sentences( ideal_summaries_filesPath[i])  
            global ex1_AP_sum,  ex2_AP_sum
            print("Ex1_summary", ex1_summary)
            #print("PR \n ", ex1_PR)
            print("\n SOMA", sum(list(ex1_PR.values())))
            if(len(ex1_summary))>0:
                ex1_AP_sum=  calculate_precision_recall_ap(ex1_summary,ideal_summary, ideal_summary_sentences,ex1_AP_sum)
            print("\n Ex2_summary", ex1_summary)
            #print("PR \n ", ex2_PR)
            print("\n SOMA", sum(list(ex2_PR.values())))
            ex2_AP_sum=  calculate_precision_recall_ap(ex2_summary,ideal_summary, ideal_summary_sentences,ex2_AP_sum)

            #break
            i+=1
    
    return len(files)  #retornas n-docs


def cosine_similarity(main_sentence,sentences_vectors ):
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
    return np.array(cosSim) 

def get_graph(sentences_vectors):
    n_sentences=len(sentences_vectors)
    tri_matrix=np.triu(np.zeros((n_sentences,n_sentences)),-1)
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        tri_matrix[node,start_index:]=cos_sim
    #print( "GRAP MEMSO \n ", tri_matrix+tri_matrix.T)
    
    graph=tri_matrix+tri_matrix.T
    
    indexes_sents_not_linked=np.where(~graph.any(axis=0)) #collum with just zeros
    #print("senteces not linked", indexes_sents_not_linked[0])
    indexes_sents=np.delete(np.arange(len(graph)),indexes_sents_not_linked)
    #print("indexes", indexes_sents)
    graph=np.delete(graph, indexes_sents_not_linked,0) #delete rows with just zeros
    graph=np.delete(graph, indexes_sents_not_linked,1) #delete collumns with just zeros
    #print("Graph agora sim", graph)
    
    return graph,indexes_sents, indexes_sents_not_linked


def get_prior(n_docs):
    weights_pos=np.arange(n_docs, 0, -1)
    print("\nnot dum 1 \n\n", (weights_pos) )
    print("\nnot dum 1 \n\n", np.sum(weights_pos) )

    return weights_pos/np.sum(weights_pos)
    
    
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
            AP_sum ):
    R=len(ideal_summary_sentences)  #documento relevante
    
    RuA = sum(1 for x in summary if x in ideal_summary_allContent)
    precision= RuA / len(summary)
    print("Precision", precision)

    correct_until_now = 0
    precisions = 0
    for i in range(len(summary)) :
        if summary[i] in ideal_summary_allContent :
            correct_until_now +=1
            precisions+= correct_until_now / (i+1)
                    
    AP_sum+=(precisions/R)
    return AP_sum 


def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    #return tfs_sent*isfs, isfs, counts_of_terms_sent
    return tfs_sent*isfs


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs


def doc_ToVectorSpace(content, isfs, counts_of_terms_sent):
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0) #summing the terms counts of each sentence
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)  #(lines=documents, cols=terms) 
    tfs_doc=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return tfs_doc*isfs
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt") 
    ideal_summaries_filesPath=get_ideal_summaries_files('TeMario/Sumarios/Extratos ideais automaticos')

    '''docs_sents_vectors, docs_vectors, docs_sentences=sents_and_docs_vectorSpace(
            'TeMario/Textos-fonte/Textos-fonte com titulo')       #podes dar como argumento o q representar: ... frases em tf ou idf; ou n-gramas
    
    '''
    n_docs=read_docs('TeMario/Textos-fonte/Textos-fonte com titulo', sentences_ToVectorSpace, get_graph,get_prior)     
    #print_results("exercise 1", (ex1_precision_sum / number_of_docs), (ex1_recall_sum / number_of_docs), (ex1_AP_sum / number_of_docs))
    print("\n exercise 1- MAP", (ex1_AP_sum / n_docs))
    print("\n exercise 2- MAP", (ex2_AP_sum / n_docs))


    '''
    sentences_vectors=sentences_ToVectorSpace(sentences)  
    graph=get_graph(sentences_vectors, 0.2)     
    print("\n Graph\n", graph)
    
    PR, sentences_not_linked = calculate_page_rank2(graph, 0.15, 50)
    print("PR \n ", PR)
    summary, summary_to_user=show_summary(PR,sentences,5)
    print("\n SOMA", sum(list(PR.values())))
    print("\n Summmary", summary)
    print(summary_to_user)
    print("--- %s seconds ---" % (time.time() - start_time))
    '''
