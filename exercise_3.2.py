# Joel Almeida		81609
# Matilde Gonçalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator

def read_file(file):
    return open(file).read()
    
def counts_and_tfs_BM25(file_content):
    k=1.2
    b=0.75
   
    vec = CountVectorizer()
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    nominator=counts_of_terms*(k+1)
    length_of_sentences_D=counts_of_terms.sum(1)
    number_sentences=counts_of_terms.shape[0]
    avgdl= sum(length_of_sentences_D)/number_sentences
    denominator=counts_of_terms+(k*(1-b+b*((length_of_sentences_D)/(avgdl))))[:, None] 
    tfs=nominator/denominator
    print("TFS values",tfs)
    return counts_of_terms,tfs
    

def sentences_ToVectorSpace_BM25(content):
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content) #as contagens e os tfs para as frases
    print(counts_of_terms_sent)

    total_number_sentences_N=counts_of_terms_sent.shape[0]
    times_word_sentences_nt=(counts_of_terms_sent!=0).sum(0)
    isfs=np.log10((total_number_sentences_N-times_word_sentences_nt+0.5)/(times_word_sentences_nt+0.5))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)
    print("ISFS values",isfs)
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
    scores_sentences_sorted = sorted(scored_sentences.items(), key=operator.itemgetter(1),reverse=True) 
    summary=sorted(scores_sentences_sorted[0:3], key=operator.itemgetter(0))  #Ponto 4 done
    return [sentences[line] for line,sim in summary], summary
    

if __name__ == "__main__":
    file_content=read_file("TeMario/Textos-fonte/Textos-fonte com título")
    sentences = nltk.sent_tokenize(file_content) #o doc dividido em frases
    sentences_vectors, isfs=sentences_ToVectorSpace(sentences)  #Ponto 1
    doc_vector=doc_ToVectorSpace(file_content, isfs)    #Ponto 2
    
    print("The vectors of the sentences:\n", sentences_vectors,"\n\n The vector of the document:\n", doc_vector)    
    
    scored_sentences=cosine_similarity(sentences_vectors,doc_vector[0])  #Ponto 3 done
    summary_to_user, summary=show_summary(scored_sentences, sentences)
    print("\n Summary: ", summary, "\n\n Result to the user",summary_to_user )  #Ponto 5 done, end!
    
    
    