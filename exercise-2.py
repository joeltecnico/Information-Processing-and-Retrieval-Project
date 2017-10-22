# Joel Almeida		81609
# Matilde Gonรงalves	82091
# Rita Ramos		    86274
#!/usr/bin/python3


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
import os
import codecs


def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content, docs_vocabulary):
    vec = CountVectorizer(vocabulary=docs_vocabulary)  #é dado o vocabulario dos documentos
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #as contagens e os tfs para as frases
    idfs=np.log10(len(counts_of_terms_sent)/((counts_of_terms_sent != 0).sum(0) +1) ) # idfs com smoothing, para não dar zero!
    return tfs_sent*idfs

def doc_ToVectorSpace(content, number_of_docs):
    vec = CountVectorizer()
    counts_of_terms_doc, tfs_doc=counts_and_tfs(content, vec)  # as contagens e tfs para o documento
    idfs=np.log10(number_of_docs/(counts_of_terms_doc != 0).sum(0))  # idfs= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_doc*idfs, vec.vocabulary_

def cosine_similarity(sentences_vectors,doc_vector):
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim


def convert_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_content=[]
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content = open(os.path.join(root, f), "rb").read().decode('iso-8859-1')
            docs_sentences[i] = nltk.sent_tokenize(file_content) #o doc dividido em frases
            docs_content.append(file_content) #vão sendo guardado os documentos todos
            i+=1
    
    number_of_docs=len(docs_content)
    docs_vectors, vocabulary=doc_ToVectorSpace(docs_content, number_of_docs)  #documentos em vector
    
    docs_sentences_vectors={}                                
    for i in range(number_of_docs):
        docs_sentences_vectors[i]=sentences_ToVectorSpace(docs_sentences[i], vocabulary)  #as frases passam a estar representadas em vectores
    
    return docs_sentences_vectors, docs_vectors, docs_sentences

    
    
def showResultsToUser(vector_of_docsSentences, vectors_of_docs):
    docs_size=len(vectors_of_docs)
    docs_cosSim={}
    for i in range(docs_size):
        docs_cosSim[i]=cosine_similarity(vector_of_docsSentences[i] , vectors_of_docs[i])
        doci_sorted_scores = sorted(docs_cosSim[i].items(), key=operator.itemgetter(1),reverse=True) #Ponto 4 done
        summary=sorted(doci_sorted_scores[0:3], key=operator.itemgetter(0)) 
        print("\n Result to the user of doc ",i,": ",[docs_sentences[i][line] for line,sim in summary])  #Ponto 5 done, end!
     
vector_of_docsSentences, vectors_of_docs, docs_sentences=convert_and_docs_ToVectorSpace('Textos-fonte-com-titulo')
#cosinesOfAllDos(vector_of_docsSentences, vectors_of_docs)
showResultsToUser(vector_of_docsSentences, vectors_of_docs)

    
    
    