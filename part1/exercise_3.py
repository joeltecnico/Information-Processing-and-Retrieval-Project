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
from nltk.tag import hmm

def simplify_tag(t):# http://www.nltk.org/howto/portuguese_en.html
    return t[t.index("+")+1:] if '+' in t else t

AP_sum = 0
precision_sum = 0
recall_sum = 0
tsents = floresta.tagged_sents()
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
arrayStopWords=nltk.corpus.stopwords.words('portuguese')
stemmer=nltk.stem.RSLPStemmer()
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(tsents)

def sentences_and_docs_ToVectorSpace(path):
    docs_sentences={}
    i=0
    docs_sentences_vectors={}
    docs_vectors={}
    
    for root, dirs, files in os.walk(path):
        for f in files:
            file_content,sentences= exercise_1.getFile_and_separete_into_sentences(os.path.join(root, f))
           
            docs_sentences_vectors[i], isfs, counts_of_terms_sent=sentences_ToVectorSpace(sentences)  
            docs_vectors[i]=doc_ToVectorSpace(file_content, isfs,counts_of_terms_sent)                
        
            docs_sentences[i] = sentences
            i+=1
    return docs_sentences_vectors,docs_vectors,  docs_sentences

def sentences_ToVectorSpace(content):
    counts_of_terms_Ngramas, sentences_words=Ngrams(content) #contagens de uni e bigramas
    counts_of_terms_Ngramas_and_nounPhrases=add_noun_phrases(counts_of_terms_Ngramas, sentences_words)                                       
    score_BM5_without_ISF=get_score_BM5_without_ISF(counts_of_terms_Ngramas_and_nounPhrases)
    isfs=get_isfs(counts_of_terms_Ngramas_and_nounPhrases)
    return score_BM5_without_ISF*isfs, isfs,counts_of_terms_Ngramas_and_nounPhrases

def doc_ToVectorSpace(content, isfs,counts_of_terms_sent ):    
    counts_of_terms=np.sum(counts_of_terms_sent, axis=0)
    counts_of_terms=np.expand_dims(counts_of_terms, axis=0)   
    score_BM5_without_ISF=get_score_BM5_without_ISF(counts_of_terms)
    return score_BM5_without_ISF*isfs

def Ngrams(doc_sentences):
    words_of_sentences = remove_stop_words(doc_sentences)
    sentences_without_stop_words= joining(words_of_sentences) 
    vec = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b') #vocabulario ser com uni e brigramas
    counts_of_terms=vec.fit_transform(sentences_without_stop_words).toarray()
    return counts_of_terms, words_of_sentences
    
def add_noun_phrases(counts_of_terms_Ngramas, sentences_words ):
    counting = {} #dict for the counts of nounphrases
    tagged_sentences = tagger.tag_sents(sentences_words)
    for i in range(len(tagged_sentences)) :
        tagged_sentence = tag_string(tagged_sentences[i])
        m = re.findall(r'(((\w+_adj )*(\w+_(n|prop) )+(\w+_(prp|conj-s|conj-c) ))?(\w+_adj )*(\w+_(n|prop) )+)+'
            , tagged_sentence, re.UNICODE)
        for group_found in m:
            noun_phrase = group_found[0].strip()
            
            if len(noun_phrase.strip().split(' ')) > 2: 
                if noun_phrase not in counting:
                    counting[noun_phrase] = [0] * len(sentences_words)
                    counting[noun_phrase][i] = 1  
                else:
                    counting[noun_phrase][i] += 1
                   
    found = np.array(list(counting.values()))   
    counts_of_terms_Ngramas_and_nounPhrases = np.concatenate((counts_of_terms_Ngramas, found.T), axis=1)
    return counts_of_terms_Ngramas_and_nounPhrases

def remove_stop_words(sentences): #para cada frase, vamos retirar as stop words
    words=[]
    for sentence in sentences:
        words_of_sentence=[word for word in re.findall(r'\w+', sentence.lower()) if word
            not in arrayStopWords]
        if len(words_of_sentence) >1: #Apos remover stopWrds, verificar se a frase não ficou vazia
            words.append(words_of_sentence)
    return words

def joining(sentences_words): #juntar as palavras de cada frase para ter em string
    file_content=[]
    for sentence in sentences_words:
        file_content.append(' '.join(word for word in sentence))
    return file_content
        
def tag_string(s) :
    sentence = ''
    for (a, b) in s:
        j = '_'.join([a, simplify_tag(b)])
        sentence = ' '.join([sentence, j])
    sentence = sentence[1:] + ' '
    return sentence



def get_score_BM5_without_ISF(counts_of_terms):
    k=2
    b=0.75
    nominator=counts_of_terms*(k+1)
    length_of_sentences_D=counts_of_terms.sum(1)
    number_sentences=len(counts_of_terms)
    avgdl= sum(length_of_sentences_D)/number_sentences
    denominator=counts_of_terms+(k*(1-b+b*((length_of_sentences_D)/(avgdl))))[:, None] 
    score_BM5_without_ISF=nominator/denominator
    return score_BM5_without_ISF

def get_isfs(counts_of_terms_sent):
    N=len(counts_of_terms_sent) #numero de frases
    nt=(counts_of_terms_sent!=0).sum(0)  #nº vezes q o termo aparece nas frases
    isfs=np.log10((N-nt+0.5)/(nt+0.5))  # inverve sentence frequency              
    return isfs

def calculate_cosine_for_the_ex(vector_of_docsSentences,  vectors_of_docs, number_of_docs):    
    cosSim_of_docs={}
    for i in range(number_of_docs):
        cosSim_of_docs[i]=exercise_1.cosine_similarity(vector_of_docsSentences[i] , vectors_of_docs[i][0])
        show_summary(cosSim_of_docs[i], i)
        
def show_summary(cosSim, id_doc):
    doc_sentences=docs_sentences[id_doc]
    summary, summary_to_user=exercise_1.show_summary(cosSim, doc_sentences, 5)
    print("\nDoc ",id_doc, ": \n\nEx3- Summary to user:", summary_to_user)
    evaluate_summaries(summary,id_doc)

def evaluate_summaries( summary, id_doc):
    ideal_summary,ideal_summary_sentences =exercise_1.getFile_and_separete_into_sentences(
            ideal_summaries_filesPath[id_doc])  
    global AP_sum, precision_sum,recall_sum
    AP_sum, precision_sum,recall_sum=  exercise_2.calculate_precision_recall_ap(summary,
            ideal_summary, ideal_summary_sentences,AP_sum, precision_sum,recall_sum)

if __name__ == "__main__":
    ideal_summaries_filesPath=exercise_2.get_ideal_summaries_files(
            'TeMario/Sumarios/Extratos ideais automaticos')
    vector_of_docsSentences,vectors_of_docs, docs_sentences =sentences_and_docs_ToVectorSpace(
            'TeMario/Textos-fonte/Textos-fonte com titulo')
    number_of_docs=len(docs_sentences)
    calculate_cosine_for_the_ex(vector_of_docsSentences, vectors_of_docs,number_of_docs)
    exercise_2.print_results("exercise 3", (precision_sum / number_of_docs),
            (recall_sum / number_of_docs),(AP_sum/number_of_docs))