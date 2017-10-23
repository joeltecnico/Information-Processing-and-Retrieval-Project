# Joel Almeida		81609
# Matilde Gonçalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator

def read_file(file):
    return open(file).read()
    
def counts_and_tfs(file_content):
    vec = CountVectorizer()
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content):
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content) #as contagens e os tfs para as frases
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_sent*isfs, isfs

def doc_ToVectorSpace(content, isfs):
    counts_of_terms_doc, tfs_doc=counts_and_tfs([content])  # as contagens e tfs para o documento
    return tfs_doc*isfs

def cosine_similarity(sentences_vectors,doc_vector):
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim


def show_summary(scored_sentences, sentences):
    #print("\n The cosine similiraty: ", scored_sentences)
    scores_sentences_sorted = sorted(scored_sentences.items(), key=operator.itemgetter(1),reverse=True) 
    #print("\n The cosine similarity sorted",scores_sentences_sorted)
    summary=sorted(scores_sentences_sorted[0:3], key=operator.itemgetter(0))  #Ponto 4 done
    #print("\n Sorted sentences to show to the user:",summary)
    return [sentences[line] for line,sim in summary]
    
    

#Results:
file_content=read_file("script1.txt")
sentences = nltk.sent_tokenize(file_content) #o doc dividido em frases

sentences_vectors, isfs=sentences_ToVectorSpace(sentences)  #Ponto 1
doc_vector=doc_ToVectorSpace(file_content, isfs)    #Ponto 2
print("The vectors of the sentences:\n", sentences_vectors,"\n\n The vector of the document:\n", doc_vector)    

scored_sentences=cosine_similarity(sentences_vectors,doc_vector[0])  #Ponto 3 done
print("\n Result to the user", show_summary(scored_sentences,sentences ))  #Ponto 5 done, end!