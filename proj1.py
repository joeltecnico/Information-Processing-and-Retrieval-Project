# Joel Almeida		81609
# Matilde Gonçalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def sentencesToVectorSpace(file):
    file_content = open(file).read()
    sentences = nltk.sent_tokenize(file_content)  #o doc dividido em frases
    vec = CountVectorizer()
    matrix_termsCounts = vec.fit_transform(sentences)  #matrix esparsa com docs e as respectivas contages dos termos (doc,termo) contagem 
    np_termsCounts=matrix_termsCounts.toarray()# igual a matrix só q em numpy para depois dar para fazer calculos
    tf_of_terms=np_termsCounts/np.max(np_termsCounts, axis=1)[:, None]  #tf= freq/max termo
    idf_of_terms=np.log10(len(np_termsCounts)/(np_termsCounts != 0).sum(0))  # idfs= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tf_of_terms*idf_of_terms

sentencesToVectorSpace("script1.txt")   #1 ponto done


