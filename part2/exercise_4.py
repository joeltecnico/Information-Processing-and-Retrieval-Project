#!/usr/bin/env python3

import urllib.request
import xml.etree.ElementTree as ET
import re
import nltk
import html2text
import exercise_1
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
from bs4 import BeautifulSoup

def splitSentences(text):
    sentences=[]
    for sentence in nltk.sent_tokenize(text):
        if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
            sentences.append(sentence)
    return sentences
    

def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()
    counts_of_terms = counts_of_terms[~np.all(counts_of_terms == 0, axis=1)]
    
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return counts_of_terms,tfs

    
def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    return tfs_sent*isfs

def getParsesPages(f): 
    news = []
    file_content = open(f, 'rb').read().decode('iso-8859-1').splitlines()
    
    for line in file_content:
        
        source_url=line.split(',')
        url=urllib.request.urlopen(source_url[1])
    
        
        root = ET.parse(url)
        for ele in root.findall(".//item"):
            news_contents = []
            
            title = parseHTML(ele.findtext('title')).replace('\n', ' ').strip()
            description = parseHTML(ele.findtext('description')).replace('\n', ' ').strip()
            
            if len(title) > 0 :
                news += splitSentences(title)
                
            if len(description) > 0 :
                news += splitSentences(description)
            
    return news
    
def parseHTML(html) :
    soup = BeautifulSoup(html, 'html5lib')

    text_parts = soup.findAll(text=True)
    text = ''.join(text_parts)
    
    return text

def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user

if __name__ == "__main__":
    news=getParsesPages('sources.txt')
    
    sentences_vectors = sentences_ToVectorSpace(news)
    graph=exercise_1.get_graph(sentences_vectors, 0.2)
    PR = exercise_1.calculate_page_rank(graph, 0.15, 50)
    summary, summary_to_user=show_summary(PR,news,5)
    for sentence in summary_to_user :
        print(sentence)
    