# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd

# # Apply TF-IDF

total_documents = 9
data = np.matrix([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 2, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0]
])
data = pd.DataFrame(data)


def tf_idf(document):
    chi = 0
    #First calculate the chi part for this docutment
    for term in document:
        if term > 0: chi +=1
    #Apply the TF-IDF weighing scheme to each element of this document column vector.
    return document*np.log(total_documents/chi)


tfidf = data.apply(lambda x: tf_idf(x), axis = 0)

print(tfidf)
#Write to latex table without columns and row indexes and format floats to 2 decimals. 
tfidf.to_latex(buf = 'tf_idf.tex',index = False, header = False, float_format="%.2f")


# # Apply Log Entropy

def log_entropy(document):
    p_sum = np.sum(document)
    p_vec = np.zeros(len(document))
    #Create vecotor with p_ij values for each term in document
    for i,term in enumerate(document):
        p_vec[i] =  (term/p_sum)
    
    #Calculate the term in swaure brackets. This is done over the entire column
    p_sum = 0
    for p_ij in p_vec:
        if p_ij != 0:
            p_sum = p_sum + (p_ij*np.log(p_ij)/np.log(total_documents))
            
    #Calcualte the final value for each term in this column. 
    res = np.zeros(len(document), dtype=float)
    for idx, term in enumerate(document):
        res[idx] = np.log(1.0+ float(term)) * (1.0+float(p_sum))
    return res


logEntropy = data.apply(lambda x: log_entropy(x), axis = 0)

print(logEntropy)
#Write to latex table without columns and row indexes and format floats to 2 decimals. 
tfidf.to_latex(buf = 'logEntropy.tex',index = False, header = False, float_format="%.2f")


