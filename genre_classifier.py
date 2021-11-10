import json
import numpy as np

# Tokenizer
from nltk.tokenize import TreebankWordTokenizer
# from sklearn.feature_extraction.text import CountVectorizer
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Binary Relevance
from sklearn.preprocessing import OneHotEncoder
# models
import sklearn
# from sklearn import svm
# from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier
# from sklearn import tree

tokenizer = TreebankWordTokenizer()
# tokenizer= CountVectorizer()

# classifier and model
model = OneVsRestClassifier(sklearn.svm.SVC(kernel='linear', probability=True))
#model=KNeighborsClassifier(n_neighbors=3,weights='distance')
# model= sklearn.svm.SVC(kernel='linear')
# model=tree.DecisionTreeClassifier()

onehot_encoder = OneHotEncoder(sparse=False)

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# One hot encoding Y values
YOneEncode = onehot_encoder.fit_transform(np.array(Y).reshape(len(Y), 1))

#The data is split keeping the class distribution same in both train and val split
X_train, X_val, Y_train, Y_val = train_test_split(X, YOneEncode, train_size=0.9, random_state=42, stratify=Y)

# vectorizing  Method
#CountVectorizer=CountVectorizer()
tfidfVectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# create embeddings
xtrainemb = tfidfVectorizer.fit_transform(X_train)
xvalemb = tfidfVectorizer.transform(X_val)

# fit model on train data
model.fit(xtrainemb.toarray(), Y_train)

# probability prediction
t = 0.5 # threshold value for propability
yPredPropa = model.predict(xvalemb.toarray())
yPredicted = (yPredPropa >= t).astype(int)

# evaluate performance
print(f1_score(Y_val, yPredicted, average="macro"))

model.fit(tfidfVectorizer.transform(X), Y)

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
X_test = test_data['X']

YtestProb = model.predict_proba(tfidfVectorizer.transform(X_test))


YtestBinary = np.zeros((len(YtestProb), 4))
for prob in range(0, len(YtestProb)):
    YtestBinary[prob][argmax(YtestProb[prob])] = 1

Y_test_pred = onehot_encoder.inverse_transform(X=YtestBinary)
# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()