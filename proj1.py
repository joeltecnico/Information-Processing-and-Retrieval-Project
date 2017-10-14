# Joel Almeida		81609
# Matilde Gonçalves	82091
# Rita Ramos			86274



import re  
from collections import Counter
import sys
import numpy as np

docMax={}
def indexInverted():    #this is the invertedIndex for each term: see lab2 ex1.  (Ex: to: 1,[0,4])
    f = open("Frases.txt")
    lines=f.readlines()
    d={}
    docIndex=1
    for line in lines:
        n_max=0
        words=re.findall(r'\w+', line.strip().lower())   # isto separa por palavras e numeros , tipo split, sem pontuacao
        counterWords=Counter(words)   #faz um dicionario term: count . Atenção que tem stop words este counter
        
        stopWords=re.findall(r'\w+', open('stopwords_en.txt').read().lower())   #devia ser o portugues eheh
        words=set(words)-set(stopWords)  #palavras sm stop words
    
        for word in words:
            n_max=max(counterWords[word], n_max)
            if word not in d:    #no caso do termo ainda não existir colocar [to: 1, [doc1, count desse term]]
                d[word]=[1, [[docIndex, counterWords[word]]]]
            else:               #se já existe o termo, então fazer append  [to: 1, [doc1, count desse term], [doc2, count desse term]]
                d[word][1].append([docIndex, counterWords[word]])                
                d[word][0]+=1
            
        docMax[docIndex]=n_max  #ver TF_max do documento
        docIndex+=1

    return d

    
indexInverted=indexInverted()
print("doc_max", docMax)
print("My indexInverted", indexInverted)


terms=[np.array(item[1])[:,0].max() for item in indexInverted.values()]
numberOfSentences= np.max(terms)


terms=[np.array(item[0])for item in indexInverted.values()]  
dfs= np.array(terms)


idfs=np.log(numberOfSentences/dfs)      ###É np?? acho q isto faz ln!!!


def representVectors():     #função que vai representar cada frase em vector, com os respectivos pesos ex: sent1=[0.5,0.6,0,0.7,etc]
    terms=list(indexInverted.keys())
    sentence_vectors={}
    doc_representation=dict.fromkeys(terms,1) 
    for i in range(1,numberOfSentences+1):
        sentence_vectors[i]=dict.fromkeys(terms,0)
    for term in terms:   #pseudo-codigo do lab2, ex3. 
        idf_t=idfs[terms.index(term)]   
        I_t=indexInverted[term][1]  #Ex: to: (0,[[doc1,coun],[doc2,count]])->  (0,[[doc1,coun],[doc2,count]])
        for (s,TF) in I_t:
            sentence_vectors[s][term]=(TF/docMax[s])*idf_t   
    return sentence_vectors,doc_representation




representVectors()



sentence_vectors,doc_representation=representVectors()



def cosineSimilarity():
    cosSim={}
    vector_doc=np.array(list(doc_representation.values()))
    for i in range(1,numberOfSentences+1):
        vector_seq=np.array(list(sentence_vectors[i].values()))
        cosSim[i]= np.dot(vector_seq,vector_doc)/(np.sqrt(  np.sum(vector_seq*vector_seq) )* np.sqrt(np.sum(vector_doc*vector_doc) ))
    return cosSim




cosineSimilarity()  



print("3 most similar doc",sorted(cosineSimilarity().values(),reverse=True)[0:3])







