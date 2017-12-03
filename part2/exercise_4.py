#!/usr/bin/env python3

import urllib.request
import xml.etree.ElementTree as ET
import re
import nltk
import exercise_1
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
from bs4 import BeautifulSoup #install BeautifulSoup4

def splitSentences(text):
    sentences=[]
    for sentence in nltk.sent_tokenize(text):
        if len(re.findall(r'\w+',sentence))>0: #check:not only pontuaction
            sentences.append(sentence)
    return sentences
    

def counts_and_tfs(file_content, vec):
    counts_of_terms=vec.fit_transform(file_content).toarray()

    sentences_without_words = np.where(~counts_of_terms.any(axis=1))

    counts_of_terms = counts_of_terms[~np.all(counts_of_terms == 0, axis=1)]
    
    tfs=counts_of_terms/np.max(counts_of_terms, axis=1)[:, None]
    return sentences_without_words,counts_of_terms,tfs

    
def sentences_ToVectorSpace(content):
    vec = CountVectorizer()
    sentences_without_words,counts_of_terms_sent, tfs_sent=counts_and_tfs(content, vec) #(lines=sent, cols=terms)
    isfs=np.log10(len(counts_of_terms_sent)/(counts_of_terms_sent != 0).sum(0))#inverve sentence frequency
    return sentences_without_words,tfs_sent*isfs

def getParsesPages(f): 
    news = []
    sources = []
    items = []
    connections = np.array([],dtype=np.int16)
    file_content = open(f, 'rb').read().decode('iso-8859-1').splitlines()
    
    for line in file_content:
        
        source_url=line.split(',')
        url=urllib.request.urlopen(source_url[1])
        
        sourceIndex = len(sources)
        sources.append(source_url[0])
        
        root = ET.parse(url)
        for ele in root.findall(".//item"):
            item = {}
            news_contents = []
            
            title = parseHTML(ele.findtext('title'))
            description = parseHTML(ele.findtext('description'))
            link = ele.findtext('link')
            
            item['title'] = title
            item['description'] = description
            item['link'] = link
            item['source'] = sourceIndex
            
            splitTitle = splitSentences(title)
            splitDescription = splitSentences(description)
            
            totalLength = len(splitTitle) + len(splitDescription)
            itemIndex = len(items)
            
            connection = np.full(totalLength, itemIndex)
            connections = np.concatenate((connections, connection), axis = 0 )
            
            items.append(item)
            
            news += splitTitle
            news += splitDescription
            
    return sources, items, connections, np.array(news)
    
def parseHTML(html) :
    soup = BeautifulSoup(html, 'html5lib')

    text_parts = soup.findAll(text=True)
    text = ''.join(text_parts).replace('\n', ' ').strip()
    
    return text

def show_summary(scored_sentences, sentences, number_of_top_sentences):
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    scores_sorted_byAppearance=sorted(scores_sorted_bySimilarity, key=operator.itemgetter(0)) 
    summary=[sentences[line] for line,sim in scores_sorted_bySimilarity] 
    summary_to_user= [sentences[line] for line,sim in scores_sorted_byAppearance]
    return summary, summary_to_user

if __name__ == "__main__":
    sources,items,connections,news=getParsesPages('sources.txt')
    sentences_without_words,sentences_vectors = sentences_ToVectorSpace(news)
    
    news = np.delete(news, sentences_without_words)
    news = np.delete(news, sentences_without_words)
    graph=exercise_1.get_graph(sentences_vectors, 0.2)
    PR = exercise_1.calculate_page_rank(graph, 0.15, 50)
    summary, summary_to_user=show_summary(PR,news,5)
    for sentence in summary_to_user :
        print(sentence)
        
    