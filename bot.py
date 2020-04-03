import pandas as pd
import numpy as np
import unittest
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer



stemmer = LancasterStemmer()
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]

    return ' '.join(stemmed_words)

le = LE()

tfv = TfidfVectorizer(min_df=1, stop_words='english')

data = pd.read_csv('UAPFaq.csv')
questions = data['Question'].values

X = []
for question in questions:
    X.append(cleanup(question))

tfv.fit(X)
le.fit(data['Class'])

X = tfv.transform(X)
y = le.transform(data['Class'])


trainx, testx, trainy, testy = train_test_split(X, y, test_size=.25, random_state=42)

model = SVC(kernel='linear')
model.fit(trainx, trainy)
print("SVC:", model.score(testx, testy))

def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))
    ixarr.sort()

    ixs = []
    for i in ixarr[-5:]:
        ixs.append(i[1])

    return ixs[::-1]
    
def chat():
    cnt = 0
    print("PRESS Q to QUIT")
    print("TYPE \"DEBUG\" to Display Debugging statements.")
    print("TYPE \"STOP\" to Stop Debugging statements.")
    print("TYPE \"TOP5\" to Display 5 most relevent results")
    print("TYPE \"CONF\" to Display the most confident result")
    print()
    print()
    DEBUG = False
    TOP5 = False
    name = input("PyBot: Hello! Welcome to OLD MUTUAL.Whats your name?\n")
    print("PyBot: Hello! Welcome to OLD MUTUAL. How may I assist you?")
    while True:
        usr = input(name+": ")

        if usr.lower() == 'yes':
            print("PyBot: Yes!")
            continue

        if usr.lower() == 'no':
            print("PyBot: No?")
            continue

        if usr == 'DEBUG':
            DEBUG = True
            print("Debugging mode on")
            continue

        if usr == 'STOP':
            DEBUG = False
            print("Debugging mode off")
            continue

        if usr == 'Q':
            print("PyBot: It was good to be of help.")
            break

        if usr == 'TOP5':
            TOP5 = True
            print("Will display 5 most relevent results now")
            continue

        if usr == 'CONF':
            TOP5 = False
            print("Only the most relevent result will be displayed")
            continue

        t_usr = tfv.transform([cleanup(usr.strip().lower())])
         
        transform_tsr = model.predict(t_usr)
        class_1 = le.inverse_transform(transform_tsr)

        class_ =  class_1[0]
        questionset = data[data['Class'] == class_]

         #model.predict() : given a trained model, predict the label of a new set of data. This method accepts one argument, the new
        #  data X_new (e.g. model.predict(X_new)), and returns the learned label for each object in the array.
       
        questionset = data[data['Class']==class_]

        if DEBUG:
            print("Question classified under category:", class_)
            print("{} Questions belong to this class".format(len(questionset)))

        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)
            cos_sims.append(sims)
            
        ind = cos_sims.index(max(cos_sims))

        if DEBUG:
            question = questionset["Question"][questionset.index[ind]]
            print("Assuming you asked: {}".format(question))

        if not TOP5:
            print("PyBot:", data['Answer'][questionset.index[ind]])
        else:
            inds = get_max5(cos_sims)
            for ix in inds:
                print("Question: "+data['Question'][questionset.index[ix]])
                print("Answer: "+data['Answer'][questionset.index[ix]])
                print('-'*50)

        print("\n"*2)
        outcome = input("Was this answer helpful? Yes/No: ").lower().strip()
        if outcome == 'yes':
            cnt = 0
        elif outcome == 'no':
            inds = get_max5(cos_sims)
            sugg_choice = input("PyBot: Do you want me to suggest you questions ? Yes/No: ").lower()
            if sugg_choice == 'yes':
                q_cnt = 1
                for ix in inds:
                    print(q_cnt,"Question: "+data['Question'][questionset.index[ix]])
                    # print("Answer: "+data['Answer'][questionset.index[ix]])
                    print('-'*50)
                    q_cnt += 1
                num = int(input("Please enter the question number you find most relevant: "))
                print("PyBot: ", data['Answer'][questionset.index[inds[num-1]]])

# class SimplisticTest(unittest.TestCase):
#     def test(self):
#         self.failUnless(True)

# if __name__ == '__main__':
#     unittest.main()
chat()
