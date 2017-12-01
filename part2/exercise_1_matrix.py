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
    return tfs_sent*isfs
    #return tfs_sent*isfs, isfs, counts_of_terms_sent


def cosine_similarity(main_sentence,sentences_vectors ):
    cosSim=[]
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,main_sentence)/(np.sqrt(
            np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(main_sentence*main_sentence) )))
    return np.array(cosSim) 

def get_graph(sentences_vectors, threshold):
    n_sentences=len(sentences_vectors)
    tri_matrix=np.triu(np.zeros((n_sentences,n_sentences)),-1)
    for node in range(n_sentences-1):
        start_index=node+1
        cos_sim=cosine_similarity(sentences_vectors[node], sentences_vectors[start_index:])
        #print("Cosine similarity",cos_sim )
        index_of_edges=np.asarray(np.where(cos_sim>=0))+start_index
        #tri_matrix[node,index_of_edges]=1
        tri_matrix[node,index_of_edges]=cos_sim
    return tri_matrix+tri_matrix.T





def calculate_page_rank(graph, damping, n_iter):
    sentences_not_linked=np.where(~graph.any(axis=0))
    print("senteces not linked", sentences_not_linked[0])
    indexes= np.arange(len(graph))
    indexes=np.delete(indexes,sentences_not_linked)
    print("indexes", indexes)
    graph=np.delete(graph, sentences_not_linked,0)
    graph=np.delete(graph, sentences_not_linked,1)
    print("Graph agora sim", graph)
    
    M=graph/np.sum(graph,axis=0)
    n_docs=len(M)
    N=np.full((n_docs, n_docs), n_docs)
    matrix=(1-damping)*M+(damping)*(1/N)
    r=np.ones((n_docs, 1))/n_docs
    #err=1
    
    print("Matrix", M)
    for i in range(n_iter) :
        r_new=matrix.dot(r)
        r=r_new

    return dict(zip(indexes, r))

def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user 
        
if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt") 
    sentences_vectors=sentences_ToVectorSpace(sentences)  
    graph=get_graph(sentences_vectors, 0.2)     
    print("\n Graph", graph)
    PR = calculate_page_rank(graph, 0.15, 50)
    print("PR \n ", PR)
    summary, summary_to_user=show_summary(PR,sentences,5)
    print("\n SOMA", sum(list(PR.values())))
    print("\n Summmary", summary)
    print(summary_to_user)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
                                
