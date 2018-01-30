# Information Processing and Retrieval

Joel Almeida
Matilde Gonçalves
Rita Ramos

This project was realized in the scope of the curriculum subject "Information Processing and Retrieval" from our Master of Computer Science in IST (Instituto Superior Técnico), Lisbon. 

The aim of this project is to perform automatic document summarization, which concerns with shortening text document(s) through an automated method, in order to create a summary with the major points of the original document(s).

The project was composed by two parts, both with 4 exercises each to be implemented in python, for which we had a score of 20/20 (for each part). We describe in more detail the project in the following section:

## Part 1 

In the 1º part of the project, we had to implement documents summarization based on 
each sentence according to the vector space model, using TF-IDF vectors. More challenging, we had then to consider the sentences as vector space representations of uni-grams along with bi-grams and noun phrases, scoring them with BM25, instead of TF-IDF. We also had to implement a more sophisticated approach, using Maximal Marginal Relevance (MRR). All these summarization methods were applied to the Portuguese documents in the TeMário dataset available at http://www.linguateca.pt/Repositorio/TeMario/.

For more info, see: Assignment_Part1.pdf

## Part 2

In the 2º part of our project, document summarization is performed with PageRank-based methods and also with a supervised learning-to-rank approach. Similar to first part of the project, the summarization methods were applied to TeMário dataset.
We then perform extractive multi-document summarization in a practical application, in particular, summarizing world news from different sources, namely the New York Times, CNN, the Washington Post, and Los Angels Times.   

For more info, see: Assignment_Part2.pdf




