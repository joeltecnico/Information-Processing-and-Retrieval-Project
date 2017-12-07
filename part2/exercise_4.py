#!/usr/bin/env python3

import urllib.request
import xml.etree.ElementTree as ET
import re
import nltk
import exercise_2
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
    return sentences_without_words,tfs_sent*isfs,counts_of_terms_sent

def getParsesPages(f): 
    news_sentences = []
    sources = []
    items = []
    connections = np.array([],dtype=np.int16)
    file_content = open(f, 'rb').read().decode('iso-8859-1').splitlines()
    
    for line in file_content:
    
        source_url=line.split(',')
        print("Reading source " + source_url[0])
        url=urllib.request.urlopen(source_url[1])
        
        sourceIndex = len(sources)
        sources.append(source_url)
        
        root = ET.parse(url)
        for ele in root.findall(".//item"):
            item = {}
            news_sentences_contents = []
            
            title = parseHTML(ele.findtext('title'))
            description = parseHTML(ele.findtext('description'))
            link = ele.findtext('link')
            date = ele.findtext('pubDate')
            
            item['title'] = title
            item['description'] = parseHTML(ele.findtext('description'))
            item['link'] = link
            item['date'] = date
            item['source'] = sourceIndex
            
            splitTitle = splitSentences(title)
            splitDescription = splitSentences(description)
            
            totalLength = len(splitTitle) + len(splitDescription)
            itemIndex = len(items)
            
            connection = np.full(totalLength, itemIndex)
            connections = np.concatenate((connections, connection), axis = 0 )
            
            items.append(item)
            
            news_sentences += splitTitle
            news_sentences += splitDescription
            
    return sources, items, connections, np.array(news_sentences)
    
def parseHTML(html) :
    soup = BeautifulSoup(html, 'html5lib')

    text_parts = soup.findAll(text=True)
    text = ''.join(text_parts).replace('\n', ' ').strip()
    
    return text

def generateHTML(scored_sentences, sentences, number_of_top_sentences):
    
    print("Generating HTML")
    scores_sorted_bySimilarity = sorted(scored_sentences.items(),
            key=operator.itemgetter(1),reverse=True)[0:number_of_top_sentences]
    top_sentences= [sentences[line] for line,sim in scores_sorted_bySimilarity]
    top_connections = [connections[line] for line,sim in scores_sorted_bySimilarity]
    
    with open('template.html','rb') as f:
        html = f.read().decode('iso-8859-1')
    
    content = "<table id=table>\n     <tr >\n    <td></td>    <td><h2>SUMMARY</h2></td>\n        <td><h2>SOURCE</td></h2>\n        <td><h2>CONTENT</h2></td>\n    </tr><hr>\n"
    for i in range(0, number_of_top_sentences) :
        top_sentence = '<p class="p1">'+top_sentences[i]+'</p>'
        source = sources[items[top_connections[i]]["source"]][0]
        source_link = sources[items[top_connections[i]]["source"]][1]
        
        source_html = '<a href="' + source_link + '">' + source + '</a>'
        
        title = items[top_connections[i]]["title"]
        description = items[top_connections[i]]["description"]
        link = items[top_connections[i]]["link"]
        date = items[top_connections[i]]["date"]
        
        content_html = '<b><a href="' + link + '">' + title + '</a></b><br>\n'
        if date is not None :
            content_html = content_html + '<p>'+date+'</p>' + '<br><br>\n'
        content_html = '<p class="p1">'+content_html + description+'</p><hr><br><br>\n'
        
        content = content + ' <tr>\n    <td>'+'<h3>'+str(i+1)+str('.')+'</h3></td>     <td id=tdnumber>' + top_sentence + '</td>\n        <td>' + source_html + '</td>\n        <td>' + '<p class="p1">'+content_html + '</p></td>\n    </tr>\n'
        
    content = content + "</table>"
    html = html.replace("%CONTENT%", content.strip())
    
    print("Saving HTML")
    html_file = open("news.html", "w")
    html_file.write(html)
    html_file.close()
    print("All done!")
    
def priorslenSents(sentences_vectors,counts_of_terms_sent):
    graph,indexes, indexes_sents_not_linked=exercise_2.get_graph(sentences_vectors)
    prior=exercise_2.get_prior_lenSents(counts_of_terms_sent, indexes_sents_not_linked)
    matrix_priors=exercise_2.get_priors(prior, indexes_sents_not_linked) #Prior/Sum_Priors
    return graph,matrix_priors,indexes

if __name__ == "__main__":
    sources,items,connections,news_sentences=getParsesPages('sources.txt')
    sentences_without_words,sentences_vectors,counts_of_terms_sent = sentences_ToVectorSpace(news_sentences)
    
    news_sentences = np.delete(news_sentences, sentences_without_words)
    connections = np.delete(connections, sentences_without_words)
    
    ex2_graph,ex2_priors,indexes=priorslenSents(sentences_vectors,counts_of_terms_sent)
    PR=exercise_2.calculate_improved_prank(ex2_graph, 0.15, 50,  ex2_priors, indexes)
    #graph=exercise_1.get_graph(sentences_vectors, 0.2)
    #PR = exercise_1.calculate_page_rank(graph, 0.15, 50)
    generateHTML(PR,news_sentences,5)
    
    
