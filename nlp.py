""" given the negative and positive reviews, train a model to predict the sentiment of a review
"""

import nltk
import glob
import os
import ssl
import string
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import LancasterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn import svm 


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


os.chdir(r'neg')
negFiles = glob.glob('*.txt')

os.chdir(r'../pos')
posFiles = glob.glob('*.txt')


vec = CountVectorizer()
neg_reviews = []
pos_reviews = []
train = []
test = []
dev = []
n = 1


def initialise(): 
    # read all the files in negFiles and store them in a list
    for file in negFiles:
        with open("../neg/" + file, 'r') as f:
            neg_reviews.append(f.read())
    # read all the files in posFiles and store them in a list
    for file in posFiles:
        with open("../pos/" + file, 'r') as f:
            pos_reviews.append(f.read())
    #Split the data provided into three data splits: Training, development and test

    # 80% of the data is used for training
    train_neg = neg_reviews[:int(len(neg_reviews)*0.8)]
    train_pos = pos_reviews[:int(len(pos_reviews)*0.8)]

    for i in range(len(train_neg)):
        train.append((train_neg[i], 0))
        train.append((train_pos[i], 1))

    # 10% of the data is used for development (validation)
    dev_neg = neg_reviews[int(len(neg_reviews)*0.8):int(len(neg_reviews)*0.9)]
    dev_pos = pos_reviews[int(len(pos_reviews)*0.8):int(len(pos_reviews)*0.9)]

    for i in range(len(dev_neg)):
        dev.append((dev_neg[i], 0))
        dev.append((dev_pos[i], 1))

    # 10% of the data is used for testing
    test_neg = neg_reviews[int(len(neg_reviews)*0.9):]
    test_pos = pos_reviews[int(len(pos_reviews)*0.9):]


    #combine negative and position in the form of a list of tuples
    for i in range(len(test_neg)):
        test.append((test_neg[i], 0))
        test.append((test_pos[i], 1))

initialise()

def featSelect(text):
    """ given a text, tokenize it and return a list of tokens
    """
    text = contractions.fix(text)
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    st = LancasterStemmer()
    #tokens = [st.stem(token) for token in tokens]
    return tokens



#feature selection on train
train = [(featSelect(text), label) for text, label in train]
trainRev = [text for text, label in train]
ngramTrain = []

for review in trainRev:
    ngramTrain.append(list(nltk.ngrams(review, n)))
trainLabel = [label for text, label in train]

#feature selection on test
test = [(featSelect(text), label) for text, label in test]
testRev = [text for text, label in test]
ngramTest = []

for review in testRev:
    ngramTest.append(list(nltk.ngrams(review, n)))


#feature selection on development
dev = [(featSelect(text), label) for text, label in dev]
devRev = [text for text, label in dev]
ngramDev = []

for review in devRev:
    ngramDev.append(list(nltk.ngrams(review, n)))


def collectVocab():
    """ collect the vocabulary from all the reviews
    """
    vocab = list(set([word for review in ngramTrain for word in review]))
    return vocab

vocab = collectVocab()

def termFreq(review):
    """ given a review, return a dictionary of term frequencies
    """
    tf = {}
    totalWords = len(review)
    for word in review:
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1
    #normalise the term frequencies
    for word in tf:
        tf[word] = tf[word] / totalWords
    return tf

xTrain = []
yTrain = []
for gram in ngramTrain:
    xTrain.append(termFreq(gram))
for text, label in train:
    yTrain.append(label)


def idf(reviews):
    """ given reviews, return a dictionary of inverse document frequencies
    idf = log(N/df) where N is the number of documents and df is the number of documents containing the term
    """
    idf = {}
    total = len(reviews)
    for review in reviews:
        words = []
        for token in review:
           words.append(token)
        for word in words:
            if word in idf:
                idf[word] += 1
            else:
                idf[word] = 1
    for word in idf:
        idf[word] = math.log(float(total / idf[word] + 1), 10)
    return idf


def tfidf(reviews):
    """ given the reviews, return a dictionary of tf-idf values
    """
    tfidf_values = []
    idfVal = idf(reviews)
    for review in reviews:
        tf = termFreq(review)
        tfidf = {term: tf[term] * idfVal[term] for term in tf}
        tfidf_values.append(tfidf)
        
    return tfidf_values

#calculate the train and test tf-idf

trainTfidf = tfidf(ngramTrain)
testTfidf = tfidf(ngramTest)
devTfidf = tfidf(ngramDev)
            

#calculate class probabilities
def classProb(yTrain):
    """ given a list of labels, calculate the class probabilities
        Note: the classes are balanced so this is redundant but 
        makes for robust code
    """
    classProb = {}
    for label in yTrain:
        if label in classProb:
            classProb[label] += 1
        else:
            classProb[label] = 1
    for label in classProb:
        classProb[label] = classProb[label] / len(yTrain)
    return classProb

#calculate the class probabilities
classProb = classProb(yTrain)

def NaiveBayes(train_tfidf, yTrain, test_tfidf):
    '''seperate tfidf values into positive and negative classes'''
    tfidf_class = {}
    for review, label in zip(train_tfidf, yTrain):
        if label in tfidf_class:
            tfidf_class[label].append(review)
        else:
            tfidf_class[label] = [review]

    '''calculate the word probabilities'''
    wordProb = {}
    for label in tfidf_class:
        wordProb[label] = {}
        for review in tfidf_class[label]:
            for word in review:
                if word in wordProb[label]:
                    wordProb[label][word] += review[word]
                else:
                    wordProb[label][word] = review[word]
        for word in wordProb[label]:
            wordProb[label][word] = wordProb[label][word] / len(tfidf_class[label])

    '''make predictions'''
    predictions = []
    for review in test_tfidf:
        negProb = 0
        posProb = 0
        for word, tfidf in review.items():
            if word in wordProb[0]:
                negProb += tfidf * wordProb[0][word]
            else:
                negProb += 0
            if word in wordProb[1]:
                posProb += tfidf * wordProb[1][word]
            else:
                posProb += 0
        negProb += math.log(classProb[0])
        posProb += math.log(classProb[1])
        if negProb > posProb:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions


def vectorise(tfidf):
    """ given the tfidf values, convert them into a matrix
        that can be read by existing classifiers
    """
    rows = []
    cols = []
    values = []
    
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    for i, review in enumerate(tfidf):
        for word, tfidf_val in review.items():
            idx = vocab_dict.get(word)
            if idx is not None:
                rows.append(i)
                cols.append(idx)
                values.append(tfidf_val)
    
    return csr_matrix((values, (rows, cols)), shape=(len(tfidf), len(vocab)))

vecTraintfidf = vectorise(trainTfidf)
vecTesttfidf = vectorise(testTfidf)
vecDevtfidf = vectorise(devTfidf)

predNaive = NaiveBayes(trainTfidf, yTrain, testTfidf)

clf = MultinomialNB()
clf.fit(vecTraintfidf, yTrain)
predNaiveSk = clf.predict(vecTesttfidf)

svmClf = svm.SVC()
svmClf.fit(vecTraintfidf, yTrain)
predSvm = svmClf.predict(vecTesttfidf)

logistic = LogisticRegression(fit_intercept = False, C=1e1)
logistic.fit(vecTraintfidf, yTrain)
predLogistic = logistic.predict(vecTesttfidf)

''' The use of two accuracy and f1score functions is redundant here
as the classes are balanced however this makes for more robust code
'''

def accuracyTest(pred):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == test[i][1]:
            correct += 1
    accuracy = correct / len(pred) * 100
    return accuracy 

def accuracyDev(pred):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == dev[i][1]:
            correct += 1
    accuracy = correct / len(pred) * 100
    return accuracy

def f1scoreTest(pred):
    """ given the predictions, calculate the f1 score
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        if pred[i] == test[i][1] and pred[i] == 1:
            tp += 1
        elif pred[i] == 1 and test[i][1] == 0:
            fp += 1
        elif pred[i] == 0 and test[i][1] == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * ((precision * recall) / (precision + recall))
    return f1score

def f1scoreDev(pred):
    """ given the predictions, calculate the f1 score
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        if pred[i] == dev[i][1] and pred[i] == 1:
            tp += 1
        elif pred[i] == 1 and dev[i][1] == 0:
            fp += 1
        elif pred[i] == 0 and dev[i][1] == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * ((precision * recall) / (precision + recall))
    return f1score

def process_user_input(review, vocab):
    """Given a review, preprocess and vectorize it."""
    tokens = featSelect(review)
    ngram_review = list(nltk.ngrams(tokens, 1))  # Change '1' for unigrams or others
    tfidf_review = tfidf([ngram_review])[0]  # Using [0] to get single result
    vec_review = vectorise([tfidf_review])  # Vectorize using existing vocab
    return vec_review

# Prediction function based on chosen model
def predict_sentiment(review, model_choice):
    """Predict sentiment (positive or negative) based on the selected model."""
    vec_review = process_user_input(review, vocab)
    
    # Predict sentiment using the selected model
    if model_choice == 'MultinomialNB':
        pred = clf.predict(vec_review)
        sentiment = "Positive" if pred[0] == 1 else "Negative"
    elif model_choice == 'SVM':
        pred = svmClf.predict(vec_review)
        sentiment = "Positive" if pred[0] == 1 else "Negative"
    elif model_choice == 'LogisticRegression':
        pred = logistic.predict(vec_review)
        sentiment = "Positive" if pred[0] == 1 else "Negative"
    else:
        sentiment = "Invalid model choice."
    
    return sentiment

# Example use case

entering = True

while entering:   
    review_text = input("Enter your review or type x to exit: ")
    if review_text == 'x':
        entering = False
        break
    model_choice = input("Choose a model (MultinomialNB, SVM, LogisticRegression): ")
    print(f"The sentiment of the review is: {predict_sentiment(review_text, model_choice)}")
