# Joel Almeida		81609
# Matilde Gonรงalves	82091
# Rita Ramos		86274


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator

def getFile_and_separete_into_sentences(f): #quer para ingles, quer para português (semelhantes)
    file_content = open(f, 'rb').read().decode('iso-8859-1')
    file_content_splitted = file_content.splitlines()
    sentences=[]
    for line in file_content_splitted:
        sentences+=nltk.sent_tokenize(line)
    return file_content,sentences 
    
def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()  #numpy array com as respectivas contages dos termos (linhas=doc or sent,cols=termo, values=contagem)
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]  #tf= freq/max termo
    return counts_of_terms,tfs

def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #as contagens numpy(linhas=sent, cols=termos) e os tfs
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))  # inverve sentence frequency= log10(nº frases do doc/ contagem das frases q tem esse termo)
    return tfs_sent*isfs, isfs, vec

def doc_ToVectorSpace(content, isfs, vec):
    counts_of_terms_doc, tfs_doc=counts_and_tfs([content], vec)  # as contagens numpy(linhas=doc, cols=termos) e os tfs
    return tfs_doc*isfs

def cosine_similarity(sentences_vectors,doc_vector):
    cosSim={}
    i=0
    for sentence_vector in sentences_vectors:
        cosSim[i]= np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector) ))
        i+=1
    return cosSim


def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sentences_sorted = sorted(scored_sentences.items(), key=operator.itemgetter(1),reverse=True)  # ordenar os scores
    summary=sorted(scores_sentences_sorted[0:number_of_top_sentences], key=operator.itemgetter(0))  #extrair as x frases mais relevantes summary= (id_sentence, score)  (Ponto 4 done)
    return [sentences[line] for line,sim in summary], summary #imprimir as frases; imprimir summary= (id_sentence, score)
    

if __name__ == "__main__":
    
    file_content, sentences=getFile_and_separete_into_sentences("script1.txt")            
                                
    sentences_vectors, isfs, vec=sentences_ToVectorSpace(sentences)  #Ponto 1
    doc_vector=doc_ToVectorSpace(file_content, isfs, vec)    #Ponto 2
    print("The vectors of the sentences:\n", sentences_vectors,"\n\n The vector of the document:\n", doc_vector)    
    
    scored_sentences=cosine_similarity(sentences_vectors,doc_vector[0])  #Ponto 3 done
    summary_to_user, summary=show_summary(scored_sentences, sentences,3)
    print("\n Summary: ", summary, "\n\n Result to the user",summary_to_user )  #Ponto 5 done, end!

   
   

