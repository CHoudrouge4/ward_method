from numpy import *
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

pca = PCA(n_components=1000)

newsgroups_train = fetch_20newsgroups(subset='train')

vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.95)

train_data = vectorizer.fit_transform(newsgroups_train.data)

train_data = train_data.todense()

pca.fit(train_data)
Y = pca.transform(train_data)

print(train_data.shape)
savetxt("news_1000.in", Y, delimiter=' ')
