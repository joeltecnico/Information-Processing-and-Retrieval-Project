#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:08:33 2017

@author: RitaRamos
"""
import urllib.request
from collections import defaultdict
import xml.etree.ElementTree as ET

import html2text


def getParsesPages(f): 
    news = defaultdict(list)
    file_content = open(f, 'rb').read().decode('iso-8859-1').splitlines()
    
    for line in file_content:
        
        source_url=line.split(',')
        url=urllib.request.urlopen(source_url[1])
    
        
        root = ET.parse(url)
        for ele in root.findall(".//item"):
            
            news_contents=[html2text.html2text(ele.findtext('title')),
                        html2text.html2text(ele.findtext('link')),
                        html2text.html2text(ele.findtext('description')) ]
            
            #text=ele.findtext('description')
            #new_without_html=html2text.html2text(text)
            #news_contents=[new_without_html]
            news[source_url[0]].append(news_contents)
    return news
        

if __name__ == "__main__":
    news=getParsesPages('sources.txt')
    print(news["NYT"])

