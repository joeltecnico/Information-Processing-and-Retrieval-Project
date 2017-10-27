#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274

import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
import os
from os import listdir
from os.path import isfile, join
import codecs
import exercise_1
import exercise_2
import re
import math
import collections


def calculate_mmr_for_the_2_exs(ex1_vector_of_docsSentences,  ex1_vectors_of_docs, ex2_vector_of_docsSentences, ex2_vectors_of_docs, number_of_docs):    
    ex1_cosSim_of_docs={}
    ex2_cosSim_of_docs={}
    l = 0.2
    print("For doc1")
    selected = collections.OrderedDict()
    for i in range(number_of_docs):
        for sentence in ex1_vector_of_docsSentences[i] :
            max_sim2 = 0
            for key in selected :
                max_sim2 = max(max_sim2, exercise_1.cosine_similarity(sentence, selected[key]))
                
            mmr = (1 - l) * exercise_1.cosine_similarity(np.array([sentence]), ex1_vector_of_docsSentences[i][0])[0] - l * max_sim2
            print('MMR Doc ', i, ' ', mmr)
        break


if __name__ == "__main__":
    ex1_vector_of_docsSentences, ex1_vectors_of_docs, docs_sentences, docs_content = exercise_2.ex1_sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
    number_of_docs=len(docs_sentences)
    ex2_vector_of_docsSentences, ex2_vectors_of_docs=exercise_2.ex2_sentences_and_docs_ToVectorSpace(docs_content, docs_sentences,number_of_docs )
    calculate_mmr_for_the_2_exs(ex1_vector_of_docsSentences, ex1_vectors_of_docs,ex2_vector_of_docsSentences, ex2_vectors_of_docs,number_of_docs)