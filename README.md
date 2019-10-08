# NLP-CRF

 The project involved implementation of Conditional Random Fields (CRF) sequence tagger for the task of Named Entity recognition. Viterbi decoding is implemented on an HMM model and we generalize the same for the CRF forward - backward algorithm and perform learning and inference on the same. The data used in this project is derived from the CoNLL 2003 Shared Task on Named Entity Recognition (Tjong Kim Sang and De Meulder, 2003).
 
 
 TO RUN THE PROJECT:
 
 python ner.py --model HMM
 
An accuracy of 76.89\% is achieved on the development set on HMM model using Viterbi decoding.
An accuracy of 84\% is achieved on the development set using CRFs with the feature set provided.
