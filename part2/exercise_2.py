#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Joel Almeida        81609
# Matilde Gonรงalves    82091
# Rita Ramos        86274

import time
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import operator
import os
import exercise_1 as ex1
import itertools


ex1_AP_sum = 0
ex2_AP_sum = 0
ex1_precision_sum=0
ex2_precision_sum=0
stop_words=nltk.corpus.stopwords.words('portuguese')

def getFile_and_sents_without_stopWords(f): 
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
    return file_without_stop_words,sentences_without_stop_words


def get_ideal_summaries_files(path):
    ideal_summaries_filesPath={}
    i=0
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.startswith('.'):
                ideal_summaries_filesPath[i]=os.path.join(root, f)
                i+=1
    return ideal_summaries_filesPath

def read_docs(path):
    i=0
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content, sentences=getFile_and_sents_without_stopWords(os.path.join(root, f))
        
            vec= CountVectorizer()
            '''Weights (Choose a weight)'''
            sents_vectors,isfs, counts_of_terms, sents_without_words=ex1.sentences_ToVectorSpace(sentences,vec)  
            #sents_vectors,isfs,counts_of_terms, sents_without_words= get_score_BM5(sentences, vec)
            #vec= CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b')
            #sents_vectors, isfs, counts_of_terms,sents_without_words= ex1.sentences_ToVectorSpace(sentences, vec)
            #sents_vectors,isfs,counts_of_terms, sents_without_words= get_score_BM5(sentences, vec)
            
            ex1_graph=ex1.get_graph(sents_vectors, 0.2)     
            ex1_PR = ex1.calculate_page_rank(ex1_graph, 0.15, 50)
            
            ex2_graph,indexes, indexes_not_linked=get_graph(sents_vectors)
            
            '''Priors (Choose a prior)'''
            #prior=get_prior_Position(len(sents_vectors),indexes_not_linked)
            #prior=get_prior_lenSents(counts_of_terms,indexes_not_linked )
            prior=get_prior_PositionAndLenSents(counts_of_terms,indexes_not_linked )
            #prior=get_prior_termsPosition(sentences,counts_of_terms, vec,indexes_not_linked)
            #prior=get_prior_NaiveBayes(counts_of_terms, indexes_not_linked)
            #doc_vector=ex1.doc_ToVectorSpace(isfs, counts_of_terms)
            #prior=get_prior_SimilarityMostRelevantSent(doc_vector, sents_vectors,indexes_not_linked ) 
            #prior=get_prior_TFIDF(doc_vector, sents_vectors,indexes_not_linked ) 
            #prior=get_prior_PositionAndLenSentsAndTFIDF(doc_vector, sents_vectors,counts_of_terms,indexes_not_linked )

            ex2_PR=calculate_improved_prank(ex2_graph, 0.15,50,  prior, indexes)
           
            sentences=np.delete(sentences, sents_without_words)
            ex1_summary, ex1_summary_to_user=ex1.show_summary(ex1_PR,sentences,5)
            ex2_summary, ex2_summary_to_user=ex1.show_summary(ex2_PR,sentences,5)
            
            ideal_summary,ideal_summary_sentences =getFile_and_sents_without_stopWords( ideal_summaries_filesPath[i])  
            
            global ex1_AP_sum,  ex2_AP_sum, ex1_precision_sum,ex2_precision_sum
            if(len(ex1_summary))>0: #exer1 sometimes has not because of thresh
                ex1_AP_sum,ex1_precision_sum =  calculate_precision_recall_ap(ex1_summary,ideal_summary, ideal_summary_sentences,ex1_AP_sum, ex1_precision_sum)
            ex2_AP_sum, ex2_precision_sum=  calculate_precision_recall_ap(ex2_summary,ideal_summary, ideal_summary_sentences,ex2_AP_sum, ex2_precision_sum)
            
            i+=1
    return len(files)


def get_graph(sentences_vectors):
    n_sentences=len(sentences_vectors)
    tri_matrix=np.triu(np.zeros((n_sentences,n_sentences)),-1)
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=ex1.cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        tri_matrix[node,start_index:]=cos_sim    
    graph=tri_matrix+tri_matrix.T
    #delete the nodes that have cosine 0 for all other sentences(not connected)
    indexes_sents_not_linked=np.where(~graph.any(axis=0))#collum with just 0s
    indexes_sents=np.delete(np.arange(len(graph)),indexes_sents_not_linked)
    graph=np.delete(graph, indexes_sents_not_linked,0)#delete rows with just 0s
    graph=np.delete(graph, indexes_sents_not_linked,1)#delete coll with just 0s
    return graph,indexes_sents, indexes_sents_not_linked


def calculate_improved_prank(graph, damping, n_iter, priors, indexes):
    transition_probs=graph/np.sum(graph,axis=0) 
    n_docs=len(transition_probs)
    #Compute Matrix -> 1-d[adjacency] + d* [priors] 
    matrix=  (((1-damping)*transition_probs).T + (damping)*(priors)).T 
    r=np.ones((n_docs, 1))/n_docs    
    for i in range(n_iter) :
        r_new=matrix.dot(r)
        r=r_new
    return dict(zip(indexes, r))

'''Prior we have implemented'''
def get_prior_TFIDF(doc_vector, sentences_vectors,indexes_not_linked ):
    priors_cos=ex1.cosine_similarity(doc_vector[0], sentences_vectors)
    return get_priors(np.expand_dims(priors_cos, axis=0), indexes_not_linked)

def get_prior_Position(n_docs,indexes_not_linked):
    priors_pos=np.arange(n_docs, 0, -1)
    return get_priors(np.expand_dims(priors_pos, axis=0), indexes_not_linked)

# dar + importancia as frases q tem + termos
def get_prior_lenSents(counts_of_terms_sent,indexes_not_linked ): 
    priors_len=(counts_of_terms_sent != 0).sum(1)    
    return get_priors(np.expand_dims(priors_len, axis=0), indexes_not_linked)

def get_prior_PositionAndLenSents(counts_of_terms_sent,indexes_not_linked ): # dar + importancia as frases q tem + termos
    priors_position_and_sentenceSize=np.arange(len(counts_of_terms_sent), 0, -1)* (counts_of_terms_sent != 0).sum(1) 
    return get_priors(np.expand_dims(priors_position_and_sentenceSize, axis=0), indexes_not_linked)


def get_prior_PositionAndLenSentsAndTFIDF(doc_vector, sentences_vectors,counts_of_terms_sent,indexes_not_linked ): # dar + importancia as frases q tem + termos
    priors_position_and_sentenceSize_TFIDF=np.arange(len(counts_of_terms_sent), 0, -1)* (counts_of_terms_sent != 0).sum(1) *ex1.cosine_similarity(doc_vector[0], sentences_vectors)
    return get_priors(np.expand_dims(priors_position_and_sentenceSize_TFIDF, axis=0), indexes_not_linked)

def get_prior_SimilarityMostRelevantSent(doc_vector, sentences_vectors,indexes_not_linked):
    cosines=ex1.cosine_similarity(doc_vector[0], sentences_vectors)
    dict_cosines=dict(enumerate(cosines))
    most_relevant_sent = sorted(dict_cosines.items(),key=operator.itemgetter(1),reverse=True)[0][0]
    priors_cos=ex1.cosine_similarity(sentences_vectors[most_relevant_sent], sentences_vectors)
    return get_priors(np.expand_dims(priors_cos, axis=0), indexes_not_linked)        

def get_prior_NaiveBayes(counts_of_terms_sent, indexes_not_linked):    
    counts_of_terms_doc = np.sum(counts_of_terms_sent, axis=0)
    lenght_doc = (len(counts_of_terms_doc))
    prob_term_given_doc = counts_of_terms_doc/lenght_doc #probability of sentences given document
    Naive_bayes_probability= np.power(prob_term_given_doc,counts_of_terms_sent)
    priors = np.prod(Naive_bayes_probability,axis=1)
    return get_priors(np.expand_dims(priors, axis=0), indexes_not_linked)


def get_prior_termsPosition(sentences,counts_of_terms, vec,indexes_not_linked):
    lists_of_words=[]
    list_unique_words=[]
    terms_position=[]
    
    for sentence in sentences:
        lists_of_words.append([word for word in re.findall(vec.token_pattern, sentence.lower())])
    
    c=list(itertools.chain.from_iterable(lists_of_words)) #append words of all sentences        
    list_unique_words=get_unique_words(c)    
    len_doc=len(list_unique_words)
    l=list(np.arange(len_doc,0,-1))
    prior=dict(zip(list_unique_words,l))
    vec_ordenado = sorted(vec.vocabulary_.items(),
    key=operator.itemgetter(1),reverse=False)
    for key in vec_ordenado:
        terms_position.append(prior[key[0]])
    counts_of_terms[counts_of_terms>1]=1 #change counts to a binary matrix
    priorFinal= counts_of_terms.dot(terms_position)
    return get_priors(np.expand_dims(priorFinal, axis=0), indexes_not_linked) 
    
def get_unique_words(words):
    seen = set()
    result = []
    for item in words:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def get_priors(non_uniform_weights, indexes_not_linked):
    #delete rows that dont belong to graph (sents are not linked to any sent)
    non_uniform_weights=np.delete(non_uniform_weights, indexes_not_linked,1) 
    return non_uniform_weights/np.sum(non_uniform_weights)
    

def calculate_precision_recall_ap(summary, ideal_summary_allContent,
                                  ideal_summary_sentences, AP_sum, precision_sum):
    
    R=len(ideal_summary_sentences)
    RuA = sum(1 for x in summary if x in ideal_summary_allContent)
    precision= RuA / len(summary)
    precision_sum+=precision

    correct_until_now = 0
    precisions = 0
    for i in range(len(summary)) :
        if summary[i] in ideal_summary_allContent :
            correct_until_now +=1
            precisions+= correct_until_now / (i+1)
                    
    AP_sum+=(precisions/R)
    return AP_sum, precision_sum


def get_score_BM5(content, vec):
    k=1.2
    b=0.75
    counts_of_terms=vec.fit_transform(content).toarray()
    sents_without_words = np.where(~counts_of_terms.any(axis=1))
    counts_of_terms = counts_of_terms[~np.all(counts_of_terms == 0, axis=1)]
    
    nominator=counts_of_terms*(k+1)
    length_of_sentences_D=counts_of_terms.sum(1)
    number_sentences=len(counts_of_terms)
    avgdl= sum(length_of_sentences_D)/number_sentences
    denominator=counts_of_terms+(k*(1-b+b*((length_of_sentences_D)/(avgdl))))[:, None] 
    score_BM5_without_ISF=nominator/denominator
    
    N=len(counts_of_terms) 
    nt=(counts_of_terms!=0).sum(0)  
    isfs=np.log10((N-nt+0.5)/(nt+0.5))    
      
    return score_BM5_without_ISF*isfs, isfs, counts_of_terms, sents_without_words


if __name__ == "__main__":
    
    ideal_summaries_filesPath=get_ideal_summaries_files('TeMario/Sumarios/Extratos ideais automaticos')
    n_docs=read_docs('TeMario/Textos-fonte/Textos-fonte com titulo')    
        
    print("\n exercise 1- MAP", (ex1_AP_sum / n_docs))
    print("\n exercise 1- Precision", (ex1_precision_sum / n_docs))

    print("\n exercise 2- MAP", (ex2_AP_sum / n_docs))
    print("\n exercise 2- Precision", (ex2_precision_sum / n_docs))