# Joel Almeida		81609
# Matilde Gonçalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def readFile(file):
    return open(file).read()
    

def counts_and_tfs(file_content):
    vec = CountVectorizer()
    counts_of_terms=vec.fit_transform(sentences).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentencesAndDocToVectorSpace(file_content):
    sentences = nltk.sent_tokenize(file_content)   #o doc dividido em frases
    counts_of_terms_sent, tfs_sent=counts_and_tfs(sentences) #as contagens e os tfs para as frases
    idfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # idfs= log10(len dos docs/ contagem dos docs q tem esse termo)
    counts_of_terms_doc, tfs_doc=counts_and_tfs(file_content)  # as contagens e tfs para o documento
    return tfs_sent*idfs, tfs_doc*idfs


#Results:
file_content=readFile("script1.txt")
sentencesVectors, docVector=sentencesVectors=sentencesToVectorSpace(file_content)  #Pont 1 e 2 done
print("The vectors of the sentences:\n", sentencesVectors,"\n\n The vector of the document:\n", docVector)    
