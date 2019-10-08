from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tfidf = TfidfVectorizer()#(stop_words=stop_words, tokenizer=tokenize, vocabulary=vocabulary)
 
# Fit the TfIdf model
sentences = ['good article.','good book.','nice','amazing article']
# tfidf.fit([sentence for sentence in sentences])
 
# # Transform a document into TfIdf coordinates
# X = tfidf.transform([sentences[0]])
X = tfidf.fit_transform(sentences)
print(X.toarray())
print(tfidf.get_feature_names(),tfidf.idf_)

df = pd.DataFrame(X.toarray(), columns = tfidf.get_feature_names())
# print(df)
# print(df.iloc[0]['article'])
# feature_names = tfidf.get_feature_names()
# dense = X.todense()
# denselist = dense.tolist()
# df = pd.DataFrame(denselist)
# s = pd.Series(df.loc['good'])
# s[s > 0].sort_values(ascending=False)[:10]