# Joel Almeida		81609
# Matilde Gonรงalves	82091
# Rita Ramos		86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
import re

def getFile_and_separete_into_sentences(f): 
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        for sentence in nltk.sent_tokenize(line):
            if len(re.findall(r'\w+',sentence))>0:  #check if the sentence is not only pontuaction
                sentences.append(sentence)
    return file_content,sentences 
    

def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #numpy array (lines=sent, cols=terms) and the tfs
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency calculation
    return tfs_sent*isfs, isfs, counts_of_terms_sent


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray() 
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs


def doc_ToVectorSpace(content, isfs, counts_of_terms_sent):
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0) #summing the terms counts of each sentence
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)  #numpy array (lines=documents, cols=terms) 
    tfs_doc=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  
    return tfs_doc*isfs


def cosine_similarity(sentences_vectors,doc_vector):
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim


def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(), key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]  # sort the scores by relevance
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0))  #sort by appearance summary= (id_sentence, score)  
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance] # Sentences in their appearance order
    return summary, summary_to_user  


if __name__ == "__main__":
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt")            
                                
    sentences_vectors, isfs, counts_of_terms_sent=sentences_ToVectorSpace(sentences)  
    doc_vector=doc_ToVectorSpace(file_content, isfs, counts_of_terms_sent)   
    print("The vectors of the sentences:\n", sentences_vectors,"\n\n The vector of the document:\n", doc_vector)    
    
    scored_sentences=cosine_similarity(sentences_vectors,doc_vector[0]) 
    summary, summary_to_user=show_summary(scored_sentences, sentences,3)

    print("\n Summary: ", summary, "\n\n Result to the user",summary_to_user )  



   
   

