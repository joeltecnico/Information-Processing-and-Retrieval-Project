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
import proj1

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


def ex1_sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_content=[]
    ex1_docs_sentences_vectors={}
    ex1_docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content = open(os.path.join(root, f), "rb").read().decode('iso-8859-1')
            sentences=nltk.sent_tokenize(file_content) #o doc dividido em frases
            
            ex1_docs_sentences_vectors[i], isfs=proj1.sentences_ToVectorSpace(sentences)   #converter frases para vectores, usando ex 1
            ex1_docs_vectors[i]=proj1.doc_ToVectorSpace(file_content, isfs)                #converter doc para vector, usando ex 1 (argument2: inverse sentence frequency)
            
            docs_sentences[i] = sentences     #vão sendo guardadas as frases para depois calcular para ex2
            docs_content.append(file_content) #vão sendo guardado os documentos todos para depois calcular-se para ex2
            i+=1
    return  ex1_docs_sentences_vectors, ex1_docs_vectors, docs_sentences,docs_content


def ex2_sentences_and_docs_ToVectorSpace(docs_content,docs_sentences,number_of_docs ):
    ex2_docs_vectors, vocabulary=doc_ToVectorSpace(docs_content, number_of_docs)  #converter docs em vector usando o ex2
    ex2_docs_sentences_vectors={}                                
    for i in range(number_of_docs):
        ex2_docs_sentences_vectors[i]=sentences_ToVectorSpace(docs_sentences[i], vocabulary)  #converter frases em vector usando o ex2 
    return ex2_docs_sentences_vectors, ex2_docs_vectors
    

def calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences,  ex1_vectors_of_docs, ex2_vector_of_docsSentences, ex2_vectors_of_docs, number_of_docs  ):    
    ex1_cosSim_of_docs={}
    ex2_cosSim_of_docs={}
    for i in range(number_of_docs):
        ex1_cosSim_of_docs[i]=proj1.cosine_similarity(ex1_vector_of_docsSentences[i] , ex1_vectors_of_docs[i][0])
        ex2_cosSim_of_docs[i]=proj1.cosine_similarity(ex2_vector_of_docsSentences[i] , ex2_vectors_of_docs[i])
        show_summary_for_the_2_exs(ex1_cosSim_of_docs[i],ex2_cosSim_of_docs[i], i)

    
def show_summary_for_the_2_exs(ex1_cosSim,ex2_cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    print("\n For Doc1: ", id_doc)
    print("\n Ex1 summary: ", proj1.show_summary(ex1_cosSim, doc_sentences))
    print("\n Ex2 summary: ", proj1.show_summary(ex2_cosSim, doc_sentences))
    #Calcular aqui o Precision e o recall
    


#Results:
ex1_vector_of_docsSentences, ex1_vectors_of_docs, docs_sentences, docs_content =ex1_sentences_and_docs_ToVectorSpace('Textos-fonte-com-titulo')
number_of_docs=len(docs_sentences)
ex2_vector_of_docsSentences, ex2_vectors_of_docs=ex2_sentences_and_docs_ToVectorSpace(docs_content, docs_sentences,number_of_docs )
calculate_cosine_for_the_2_exs(ex1_vector_of_docsSentences, ex1_vectors_of_docs,ex2_vector_of_docsSentences, ex2_vectors_of_docs,number_of_docs)

#Comentários:
    # o output do ficheiro 1 (proj1.py) está a ser chamado, quando devia estar a correr apenas as funcoes que chamei)
    
    
    