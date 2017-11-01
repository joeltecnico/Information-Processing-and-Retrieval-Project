#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import exercise_1
import exercise_2
import re
from nltk.corpus import floresta

AP_sum = 0
precision_sum = 0
recall_sum = 0
tsents = floresta.tagged_sents()
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
tagger0 = nltk.DefaultTagger('n')
tagger1 = nltk.UnigramTagger(tsents, backoff=tagger0)
arrayStopWords=nltk.corpus.stopwords.words('portuguese')
stemmer=nltk.stem.RSLPStemmer()


def sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_sentences_vectors={}
    docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content,sentences= exercise_1.getFile_and_separete_into_sentences(os.path.join(root, f))
           
            docs_sentences_vectors[i], isfs, vocabulary, counts_of_terms_sent=sentences_ToVectorSpace(sentences)   #converter frases para vectores, usando ex 1
            docs_vectors[i]=doc_ToVectorSpace(file_content, isfs, vocabulary,counts_of_terms_sent)                #converter doc para vector, usando ex 1 (argument2: inverse sentence frequency)
        
            docs_sentences[i] = sentences     #vão sendo guardadas as frases para depois calcular para ex2 
            i+=1
    return docs_sentences_vectors,docs_vectors,  docs_sentences

def sentences_ToVectorSpace(content):
    counts_of_terms_Ngramas,vec, sentences_words=Ngrams(content) #as contagens e os tfs para as frases
    counts_of_terms_Ngramas_and_nounPhrases=add_noun_phrases(counts_of_terms_Ngramas, vec, sentences_words)                                       
    score_BM5_without_ISF=get_score_BM5_without_ISF(counts_of_terms_Ngramas_and_nounPhrases)
    isfs=get_isfs(counts_of_terms_Ngramas_and_nounPhrases)
    return score_BM5_without_ISF*isfs, isfs, vec.vocabulary_,counts_of_terms_Ngramas_and_nounPhrases

def doc_ToVectorSpace(content, isfs,docs_vocabulary,counts_of_terms_sent ):    
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0)
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)   
    score_BM5_without_ISF=get_score_BM5_without_ISF(counts_of_terms)
    return score_BM5_without_ISF*isfs


def Ngrams(doc_sentences):
    words_of_sentences = remove_stop_words(doc_sentences)#words separadas e removida
    sentences_without_stop_words= joining(words_of_sentences)
    
    vec = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b')  #vocabulario das frases do documento com unigrama e brigrama
    counts_of_terms=vec.fit_transform(sentences_without_stop_words).toarray()  #numpy array com as respectivas contages dos termos (linha=doc,col=termo, value=contagem)
    return counts_of_terms, vec, words_of_sentences
    
def add_noun_phrases(counts_of_terms_Ngramas, vec, sentences_words ):
    len_of_vocabulary=len(vec.vocabulary_)
    counting = {}
    tagged_sentences = tagger1.tag_sents(sentences_words)
    for i in range(len(tagged_sentences)) :
        tag_sentence = tag_string(tagged_sentences[i])
        m = re.findall(r'((( \w+_JJ)*( \w+_NN)+( \w+_IN))?( \w+_JJ)*( \w+_NN)+)+', tag_sentence, re.UNICODE)
        for group_found in m:
            group = group_found[0].strip()
            
            if len(group.strip().split(' ')) > 2:
                
                if group not in counting:
                    counting[group] = [0] * len(sentences_words)
                    counting[group][i] = 1
                    vec.vocabulary_[group]=len_of_vocabulary
                    len_of_vocabulary+=1
                    
                else:
                    counting[group][i] += 1
                   
    
    found = np.array(list(counting.values()))   
    counts_of_terms_Ngramas_and_nounPhrases = np.concatenate((counts_of_terms_Ngramas, found.T), axis=1)
    
    #counts_of_terms_Ngramas_and_nounPhrases=np.delete(counts_of_terms_Ngramas_and_nounPhrases, np.argwhere( (counts_of_terms_Ngramas_and_nounPhrases!=0).sum(axis=0) < 2), axis=1)
    #counts_of_terms_Ngramas_and_nounPhrases=np.delete(counts_of_terms_Ngramas_and_nounPhrases, np.argwhere( (counts_of_terms_Ngramas_and_nounPhrases!=0).sum(axis=1) ==0), axis=0)
    
    #counts_of_terms_Ngramas_and_nounPhrases=np.delete(counts_of_terms_Ngramas_and_nounPhrases,np.argwhere( (((counts_of_terms_Ngramas_and_nounPhrases!=0).sum(axis=0))/len(counts_of_terms_Ngramas_and_nounPhrases))>0.9), axis=1)
    return counts_of_terms_Ngramas_and_nounPhrases

def remove_stop_words(sentences):
    words=[]
    for sentence in sentences:
        words_of_sentence=[word for word in re.findall(r'\w+', sentence.lower()) if word not in arrayStopWords]
        if len(words_of_sentence) >1: #After removing stop words, check that the sentence is not empty
            words.append(words_of_sentence)
    return words

def joining(sentences_words):
    file_content=[]
    for sentence in sentences_words:
        file_content.append(' '.join(word for word in sentence)) 
    return file_content


def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t
      
def translate_tag(t):
    t = simplify_tag(t)
    if t == 'n' or t == 'prop':
        return 'NN'
    elif t == 'adj':
        return 'JJ'
    elif t == 'prp' or t == 'conj-s' or t == 'conj-c':
        return 'IN'
    else :
        return t
        
def tag_string(s) :
    sentence = ''
    for (a, b) in s:
        j = '_'.join([a, translate_tag(b)])
        sentence = ' '.join([sentence, j])
    return sentence


def get_score_BM5_without_ISF(counts_of_terms):
    k=1.2
    b=0.75
    nominator=counts_of_terms*(k+1)
    length_of_sentences_D=counts_of_terms.sum(1)
    number_sentences=counts_of_terms.shape[0]
    avgdl= sum(length_of_sentences_D)/number_sentences
    denominator=counts_of_terms+(k*(1-b+b*((length_of_sentences_D)/(avgdl))))[:, None] 
    score_BM5_without_ISF=nominator/denominator
    return score_BM5_without_ISF

def get_isfs(counts_of_terms_sent):
    total_number_sentences_N=counts_of_terms_sent.shape[0]
    times_word_sentences_nt=(counts_of_terms_sent!=0).sum(0)
    isfs=np.log10((total_number_sentences_N-times_word_sentences_nt+0.5)/(times_word_sentences_nt+0.5))  # inverve sentence frequency= log10(len dos docs/ contagem dos docs q tem esse termo)                   
    return isfs


def calculate_cosine_for_the_ex(vector_of_docsSentences,  vectors_of_docs, number_of_docs):    
    cosSim_of_docs={}
    for i in range(number_of_docs):
        cosSim_of_docs[i]=exercise_1.cosine_similarity(vector_of_docsSentences[i] , vectors_of_docs[i][0])
        show_summary(cosSim_of_docs[i], i)
        

def show_summary(cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    summary, scores_sentences=exercise_1.show_summary(cosSim, doc_sentences, 5)
    evaluate_summaries(summary,id_doc)


def evaluate_summaries( summary, id_doc):
    ideal_summary,ideal_summary_sentences =exercise_1.getFile_and_separete_into_sentences(ideal_summaries_filesPath[id_doc])  
    global AP_sum, precision_sum,recall_sum
    AP_sum, precision_sum,recall_sum=  exercise_2.calculate_precision_recall_ap(summary, ideal_summary, ideal_summary_sentences,AP_sum, precision_sum,recall_sum)

    
if __name__ == "__main__":
    ideal_summaries_filesPath=exercise_2.get_ideal_summaries_files('TeMario/Sumários/Extratos ideais automáticos')
    vector_of_docsSentences,vectors_of_docs, docs_sentences =sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
    number_of_docs=len(docs_sentences)
    calculate_cosine_for_the_ex(vector_of_docsSentences, vectors_of_docs,number_of_docs)
    exercise_2.print_results("exercise 3", (precision_sum / number_of_docs),(recall_sum / number_of_docs),(AP_sum/number_of_docs))