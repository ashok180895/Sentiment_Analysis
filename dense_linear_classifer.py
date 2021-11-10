import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac * len(Xr))
val_end = int((train_frac + val_frac) * len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w: v for w, v in zip(data["words"], data["vectors"])}

# initialize a tokenizer
tokenizer = TreebankWordTokenizer()


# convert a document into a vector
def document_to_vector(corpus):
    """Takes corpus and turns it into a vector array
    by aggregating its word temp.
    Here Each review is tokenized first and checked in the w2v , and vector is added.
    for missing words synonyms are found using wordnet.
    Again each word in synonym list is checked in w2v and first occurence of synonym's vector is added.
    Args:
        corpus (list) : The corpus consists of all reviews

    Returns:
        np.array: The word vector this will be 300 dimension.
    """
    vec = []
    # Tokenize the input documents
    for review in corpus:
        toks = tokenizer.tokenize(review)
        temp = []
        synonyms = []
        for i in toks:
            if i in w2v:
                temp.append(w2v[i])
            else:
                for synset in wn.synsets(i[:-1]):  # for words ending "." like animal.
                    lem = [lemma.name() for lemma in synset.lemmas()]#list of Synonyms
                for j in lem:
                    if j in w2v:
                        synonyms.append(j)#after occurence of synonym in the dictionary.
                        break
        for i in synonyms:
            temp.append(w2v[i])
        # Aggregate the temp of words in the input document, Aggregation is done by computing mean
        vec.append(np.mean(np.array(temp), axis=0))
    return vec


# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # TODO: convert each of the training documents into a vector
    XtrVector = document_to_vector(Xtr)
    # TODO: train the logistic regression classifier
    log_reg = LogisticRegression(C=C,max_iter=100)
    model = log_reg.fit(XtrVector, Ytr)
    return model


# fit a linear model
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    # TODO: convert each of the testing documents into a vector
    XtstVector = document_to_vector(Xtst)

    # TODO: test the logistic regression classifier and calculate the accuracy
    score = model.score(XtstVector, Ytst)
    return score


# TODO: search for the best C parameter using the validation set
#finding the best c values with max accuracy on the validation set
cValues = [0.1,4,8,16,25,32]
scores=[]
for i in cValues:
    mod=fit_model(X_train,Y_train,i)
    score=test_model(mod,X_val,Y_val)
    scores.append(score)
print(scores,cValues)
cBest_idx=scores.index(max(scores))
cBest=cValues[cBest_idx]
print("The best c Parameter is ::",cBest)
# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result

# computing on combined train and Val sets , Testing on test sets
training_frac = 0.8
training_end = int(training_frac * len(Xr))

# store the train test splits
X_tr = Xr[0:training_end]
Y_tr = Yr[0:training_end]
X_ts = Xr[training_end:]
Y_ts = Yr[training_end:]
final_model=fit_model(X_tr,Y_tr,cBest)
FinalAccuracy = test_model(final_model,X_ts,Y_ts)
print("Final Test Accuracy :::",FinalAccuracy)
