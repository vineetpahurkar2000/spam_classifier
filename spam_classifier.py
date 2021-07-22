# -*- coding: utf-8 -*-
"""
Created on monday June 7 18:55:52 2021

@author: vineet pahurkar
"""

import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#function to read emails
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

#function to append to our dataframe called "data"(initialized below)
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

#change the path of spam and ham folders to where you have them on your system 
data = data.append(dataFrameFromDirectory('c:/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('c:/emails/ham', 'ham'))

#CountVectorizer to split up each message into its list of words, and throw that into a MultinomialNB classifier.
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

#check this classifier with any example
mail = input("Enter the title/a few lines from your e-mail: ")
example=[mail]
example_counts = vectorizer.transform(example)
predictions = classifier.predict(example_counts)
print(predictions)
