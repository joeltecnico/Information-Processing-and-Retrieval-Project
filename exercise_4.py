#!/usr/bin/python3

# Joel Almeida		81609
# Matilde Goncalves	82091
# Rita Ramos        86274


import numpy as np
import exercise_1
import exercise_2
from collections import defaultdict
import copy

AP_sum_MMR = 0
precision_sum_MMR= 0
recall_sum_MMR = 0
AP_sum_Top5 = 0
precision_sum_Top5=0
recall_sum_Top5= 0


def cosine_similarity_asNumpy(sentences_vectors,doc_vector):
    cosSim=[]
    i=0
    for sentence_vector in sentences_vectors:
        cosSim.append( np.dot(sentence_vector,doc_vector)/(np.sqrt(  np.sum(sentence_vector*sentence_vector) )* np.sqrt(np.sum(doc_vector*doc_vector)) ))
        i+=1
    return np.array(cosSim)

def calculate_mmr_oficial(vector_of_docsSentences,vectors_of_docs, number_of_docs, docs_sentences):
    summary_of_docs=defaultdict(list)
    y=0
    
    for summary_level in range(5):
        for i in range(number_of_docs):
            vector_of_docSentence=vector_of_docsSentences[i]
            cosSim_sents_vs_doc=cosine_similarity_asNumpy(vector_of_docSentence , vectors_of_docs[i][0])
            cosSim_sents_vs_selectedSents=np.zeros(len(vector_of_docSentence))            
            for previous_sentence_selected in summary_of_docs[i]:
                cosSim_sents_vs_selectedSents+=cosine_similarity_asNumpy(vector_of_docSentence , previous_sentence_selected[0] )
            
            MMR_of_sentences=(1-y)*cosSim_sents_vs_doc-y*(cosSim_sents_vs_selectedSents)
            
            #Select sentence with highest MMR
            index_of_sentence_selected=np.argmax(MMR_of_sentences)  
            vector_of_sentence_selected=vector_of_docSentence[index_of_sentence_selected]
            string_of_sentence_selected=docs_sentences[i][index_of_sentence_selected]
        
            summary_of_docs[i].append((vector_of_sentence_selected,string_of_sentence_selected ))  #Append selected sentence as its (vector, string)

            #Now that the string was selected, remove it so that it will not be chosen next time 
            vector_of_docsSentences[i]=np.delete(vector_of_docSentence, index_of_sentence_selected, 0)
            docs_sentences[i].remove(string_of_sentence_selected)
            
    return summary_of_docs
    

def show_summary_ordered(summary_of_docs, number_of_docs):
    for i in range(number_of_docs):
        doc_sentences=docs_sentences[i]
        indexs_of_selected_sentences=[]
        for selected_sentence in summary_of_docs[i]:
            indexs_of_selected_sentences.append(doc_sentences.index(selected_sentence[1]))
        summary=sorted(indexs_of_selected_sentences)

        summary_to_user_MMR=[doc_sentences[line] for line in summary]
        summary_to_user_Top5=doc_sentences[0:5]
        
        evaluate_summaries(summary_to_user_MMR, summary_to_user_Top5, i)

        
def evaluate_summaries( summary_to_user, summary_to_user_Top5, id_doc):
    ideal_summaries_filesPath=exercise_2.get_ideal_summaries_files('TeMario/Sumários/Extratos ideais automáticos')
    ideal_summary,ideal_summary_sentences =exercise_1.getFile_and_separete_into_sentences(ideal_summaries_filesPath[id_doc])  
    global AP_sum_MMR, precision_sum_MMR,recall_sum_MMR,AP_sum_Top5, precision_sum_Top5,recall_sum_Top5
    AP_sum_MMR, precision_sum_MMR,recall_sum_MMR=  exercise_2.calculate_precision_recall_ap(summary_to_user, ideal_summary, ideal_summary_sentences,AP_sum_MMR, precision_sum_MMR,recall_sum_MMR)
    AP_sum_Top5, precision_sum_Top5,recall_sum_Top5=  exercise_2.calculate_precision_recall_ap(summary_to_user_Top5, ideal_summary, ideal_summary_sentences,AP_sum_Top5, precision_sum_Top5,recall_sum_Top5)



if __name__ == "__main__":
    vector_of_docsSentences, vectors_of_docs, docs_sentences, docs_content = exercise_2.ex1_sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
    number_of_docs=len(docs_content)
    summary_of_docs=calculate_mmr_oficial(vector_of_docsSentences,vectors_of_docs, number_of_docs, copy.deepcopy(docs_sentences))
    show_summary_ordered(summary_of_docs,number_of_docs )
    exercise_2.print_results("exercise 4- alternative MMR", (precision_sum_MMR / number_of_docs),(recall_sum_MMR  / number_of_docs),(AP_sum_MMR /number_of_docs))
    exercise_2.print_results("exercise 4 - alternative top 5", (precision_sum_Top5 / number_of_docs),(recall_sum_Top5 / number_of_docs),(AP_sum_Top5/number_of_docs))

            
    
                
            
            
    
         
            
            
    
        
        
    

    


'''JOEL PART

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
    vector_of_docsSentences, vectors_of_docs, docs_sentences, docs_content = exercise_2.ex1_sentences_and_docs_ToVectorSpace('TeMario/Textos-fonte/Textos-fonte com título')
    number_of_docs=len(docs_sentences)
    ex2_vector_of_docsSentences, ex2_vectors_of_docs=exercise_2.ex2_sentences_and_docs_ToVectorSpace(docs_content, docs_sentences,number_of_docs )
    calculate_mmr_for_the_2_exs(ex1_vector_of_docsSentences, ex1_vectors_of_docs,ex2_vector_of_docsSentences, ex2_vectors_of_docs,number_of_docs)
    '''