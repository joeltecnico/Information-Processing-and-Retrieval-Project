
# coding: utf-8

# In[34]:

import re
from collections import Counter
import sys
import numpy as np
def index():
    f = open("script1.txt")
    lines=f.readlines()
    d={}
    docIndex=1
    
    for line in lines:
        words=re.findall(r'\w+', line.strip().lower())
        counterWords=Counter(words)
        words=set(words)
        for word in words:
            if word not in d:
                d[word]=[1, [[docIndex, counterWords[word]]]]
            else:
                d[word][1].append([docIndex, counterWords[word]])                
                d[word][0]+=1
        
        docIndex+=1
    return d
    
indexInverted=index();
print("My indexInverted", indexInverted)


# In[2]:

terms=[np.array(item[1])[:,0].max() for item in indexInverted.values()]
numberOfSentences= np.max(terms)


# In[3]:

terms=[np.array(item[0])for item in indexInverted.values()]
dfs= np.array(terms)


# In[5]:

idfs=np.log(numberOfSentences/dfs)      ###É np?? acho q isto faz ln!!!


# In[7]:

def representVectors():
    terms=list(indexInverted.keys())
    sentence_vectors={}
    doc_representation=dict.fromkeys(terms,1)
    for i in range(1,numberOfSentences+1):
        sentence_vectors[i]=dict.fromkeys(terms,0)
    for term in terms:
        idf_t=idfs[terms.index(term)]
        I_t=indexInverted[term][1]
        for (s,TF) in I_t:
            sentence_vectors[s][term]=TF*idf_t      ##TF é normalizado??

    return sentence_vectors,doc_representation


# In[8]:

representVectors()


# In[24]:

sentence_vectors,doc_representation=representVectors()


# In[25]:

from sklearn.metrics.pairwise import cosine_similarity
def myCosineSimilarity():
    cosSim={}
    for i in range(1,numberOfSentences+1):
         cosSim[i]=cosine_similarity(list(sentence_vectors[i].values()),list(doc_representation.values()))
    return cosSim


# In[27]:

myCosineSimilarity()


# In[28]:

def cosineSimilarity():
    cosSim={}
    vector_doc=np.array(list(doc_representation.values()))
    for i in range(1,numberOfSentences+1):
        vector_seq=np.array(list(sentence_vectors[i].values()))
        cosSim[i]= np.dot(vector_seq,vector_doc)/(np.sqrt(  np.sum(vector_seq*vector_seq) )* np.sqrt(np.sum(vector_doc*vector_doc) ))
    return cosSim


# In[23]:

cosineSimilarity()


# In[29]:

print("3 most similar doc",sorted(cosineSimilarity().values(),reverse=True)[0:3])


# In[ ]:




# In[31]:

set(["ola","adeus", "pois"])-set(["ola"])


# In[32]:

import re
from collections import Counter
import sys
import numpy as np
def index():
    arrayWords=re.findall(r'\w+', open('script1.txt').read().lower())
    indexInverted= Counter(arrayWords)
    f = open("script1.txt")
    lines=f.readlines()
    d={}
    docIndex=1
    
    for line in lines:
        
        words=re.findall(r'\w+', line.strip().lower())
        counterWords=Counter(words)
        
        stopWords=re.findall(r'\w+', open('stopwords_en.txt').read().lower())   #devia ser o portugues eheh
        words=set(words)-set(stopWords)
        for word in words:
            if word not in d:
                d[word]=[1, [[docIndex, counterWords[word]]]]
            else:
                d[word][1].append([docIndex, counterWords[word]])                
                d[word][0]+=1
        
        docIndex+=1
    return d
    
indexInverted=index();
print("My indexInverted", indexInverted)


# In[ ]:



