# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
from nltk import bigrams
from collections import Counter
import re

def read_file(file):
    return open(file).read()

def sentences_separation(file_content):
    sentences = nltk.sent_tokenize(file_content) #o doc dividido em frases
    return sentences

def words_separation(sentences):
    words=[]
    for t in sentences:
        words.append(nltk.word_tokenize(t))
    print ("woooords",words)
    return words

def words_separation2(sentences):
    words=[]
    for t in sentences:
        print ("TTTTTT",t)
        words.append(re.findall(r'\w+',t.strip().lower()))
        
    print ("frasesssssssss",sentences)
    print ("woooords",words)
    return words

def counts_and_tfs_biGrams(file_content):
    bi_grams=[]
    for i in file_content :              
        bi_grams.append(list(bigrams(i))) #diz os bi-grams
   # vec = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b')
    vec = CountVectorizer()
    counts_of_terms=vec.fit_transform(bi_grams).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
   
   # tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    print ("bigramassa",bi_grams) 
   # return counts_of_terms,tfs
    
def sentences_ToVectorSpace(content):
    print("connnteg",content)
    counts_of_terms_sent, tfs_sent=counts_and_tfs_biGrams(content) #as contagens e os tfs para as frases
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)
    return tfs_sent*isfs, isfs

def doc_ToVectorSpace(content, isfs):
    counts_of_terms_doc, tfs_doc=counts_and_tfs_biGrams([content])  # as contagens e tfs para o documento
    return tfs_doc*isfs

def cosine_similarity(sentences_vectors,doc_vector):
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim


def show_summary(scored_sentences, sentences):
    scores_sentences_sorted = sorted(scored_sentences.items(), key=operator.itemgetter(1),reverse=True) 
    summary=sorted(scores_sentences_sorted[0:3], key=operator.itemgetter(0))  #Ponto 4 done
    return [sentences[line] for line,sim in summary], summary
    

if __name__ == "__main__":
    file_content=read_file("script1.txt")
    sentences=sentences_separation(file_content)
    print("fraaasssses:",(sentences))
    words=words_separation2(sentences)
    sentences_vectors, isfs=sentences_ToVectorSpace(words)  #Ponto 1
    
   # doc_vector=doc_ToVectorSpace(file_content, isfs)    #Ponto 2
   # print("The vectors of the sentences:\n", sentences_vectors,"\n\n The vector of the document:\n", doc_vector)    
    
    #scored_sentences=cosine_similarity(sentences_vectors,doc_vector[0])  #Ponto 3 done
   # summary_to_user, summary=show_summary(scored_sentences, sentences)
   # print("\n Summary: ", summary, "\n\n Result to the user",summary_to_user )  #Ponto 5 done, end!

   
   

