#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Joel Almeida        81609
# Matilde Gonรงalves    82091
# Rita Ramos        86274

import time
start_time = time.time()

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
import re
import operator

def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
                sentences.append(sentence)
    return file_content,sentences 


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs

    
def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    return tfs_sent*isfs, isfs, counts_of_terms_sent

def doc_ToVectorSpace(content, isfs, counts_of_terms_sent):
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0) #summing the terms counts of each sentence
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)  #(lines=documents, cols=terms) 
    tfs_doc=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  
    return tfs_doc*isfs


def cosine_similarity(main_sentence,sentences_vectors ):  #retorna numpy
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append(cosine_between(sentence_vector, main_sentence))
        '''
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
        '''
    return np.array(cosSim) 

def cosine_between(sent1, sent2):
    return ( np.dot(sent1,sent2)/(np.sqrt(
            np.sum(sent1*sent1) )* np.sqrt(np.sum(sent2*sent2) )))


def cosine_similarity2(sentences_vectors,doc_vector):  #isto retorna em dict
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim



def prior_prob(sentences_vectors):  ###PERGUNTAR SE ´E PARA TODOS OS PJ
    n_docs=len(sentences_vectors)
    max_prior=n_docs
    nodes_prior={}
    #sum_priors=sum(   list(range(n_docs, 0,-1)))

    for i in range(n_docs):
        nodes_prior[i]=(max_prior-i)
        #/sum_priors
            
    return nodes_prior
    


def get_graph(sentences_vectors, threshold):
    graph = defaultdict(list)
    n_sentences=len(sentences_vectors)
    
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        index_of_edges=np.asarray(np.where(cos_sim>threshold))+start_index
        if len(index_of_edges[0])>0:
            graph[node] += list(index_of_edges[0])
            for i in index_of_edges[0] :
                graph[i].append(node)
    return graph


def calculate_page_rank(graph, d, n_iter):
    n_docs = len(graph)
    jump_random = d / n_docs
    prob_not_dumping = 1 - d
    print("GRAPH", graph)
    PR = dict.fromkeys(graph.keys(), 1/n_docs)
    print("PR", PR)
    for i in range(n_iter) :
        PR_new= {}
        for node in graph :
            sum_links = 0
            for link in graph[node] :
                sum_links += PR[link]/len(graph[link])
            PR_new[node] = jump_random + (prob_not_dumping * sum_links)
        print("PR_new", PR_new)
        PR=PR_new
        print("PR",PR[0])
    return PR

def calculate_improved_PR(graph, d, n_iter, nodes_prior, sentences_vectors):
    n_docs = len(graph)
    #jump_random = d / n_docs
    prob_not_dumping = 1 - d
    print("GRAPH", graph)
    PR = dict.fromkeys(graph.keys(), 1/n_docs)
    print("PR", PR)
    for i in range(n_iter) :
        PR_new= {}
        for node in graph :
            sum_links = 0
            sum_priors=0
            
            for link in graph[node] :

                pi_weight=cosine_between(sentences_vectors[node], sentences_vectors[link])            
                sum_pk_weights=0
                for k_link in graph[link]:
                    sum_pk_weights+=cosine_between(sentences_vectors[link], sentences_vectors[k_link])
                sum_links += (PR[link]*pi_weight)/(sum_pk_weights)

            for node_prior in graph:
                sum_priors+=nodes_prior[node_prior]

            PR_new[node] = d* (nodes_prior[node]/sum_priors) + (prob_not_dumping * sum_links)
        #print("IMPROVEDDD", PR_new)
        PR=PR_new
        #print("PR",PR[0])
    return PR


    
def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user  
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("TeMario/Textos-fonte/Textos-fonte com titulo/po96ju13-a.txt") 
    sentences_vectors, isfs, counts_of_terms_sent=sentences_ToVectorSpace(sentences)
    doc_vector=doc_ToVectorSpace(file_content, isfs, counts_of_terms_sent)   
    #priors=prior_prob(sentences_vectors)
    
    priors=cosine_similarity2(sentences_vectors,doc_vector[0])
    graph=get_graph(sentences_vectors, 0.2)
    #PR = calculate_page_rank(graph, 0.15, 1)
    PR= calculate_improved_PR(graph, 0.15, 50,priors,sentences_vectors )
    print("VALUE", sum(list(PR.values())))
    summary, summary_to_user=show_summary(PR,sentences,5)
    
    print(summary_to_user)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    