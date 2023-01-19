from joblib import load
from flask import Flask, render_template, request, url_for

import pandas as pd
import numpy as np
from flair.models import TextClassifier
from flair.data import Sentence

import spacy
import nltk
import json
from multi_rake import Rake
from nltk.corpus import stopwords

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


nltk.download('stopwords')
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)


@app.route('/')
def sent_seg(i_final):

    l1 = i_final.split(".")
    sent = []
    for i in l1:
        l2 = i.split(",")
        for j in l2:
            l3 = j.split(" but ")
            for k in l3:
                l4 = k.split(" and ")
                for l in l4:
                    sent += l.split(" or ")

    return (sent)


def sent_analyze(i):
    classifier = TextClassifier.load('en-sentiment')
    senti = Sentence(i)
    classifier.predict(senti)
    vs = analyzer.polarity_scores(i)
    # print(str(senti.labels))
    if "POSITIVE" in str(senti.labels) or (vs['pos'] > 0 and vs['neg'] == 0):
        return i
    else:
        return None

# Removing Stopwords and other chracters


def rem(s, c):
    d = s.split(c)
    s = "".join(d)
    return s


def remove_others(txt):
    others = [">", "<", "?", "}", "{", "[", "]", ";", "•", "*", "\n", "(", ")"]
    txtf = txt
    for i in others:
        if (i in txtf):
            txtf = rem(txtf, i)
    return txtf


def remove_stopwords(txt):
    stop_words = list(stopwords.words('english'))
    words = txt.split()
    buff = []
    for w in words:
        if (w not in stop_words):
            buff.append(w)
    txt1 = " ".join(buff)
    return txt1


def spacy_implement(test):
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    # Process whole documents

    doc = nlp(test)
    spacy_nouns = []
    for chunk in doc.noun_chunks:
        # print(chunk)
        spacy_nouns.append(chunk.text)

    # Analyze syntax
    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    return spacy_nouns

    # Find named entities, phrases and concepts
    # for entity in doc.ents:
    # print(entity.text, entity.label_)


rake = Rake()


def keywords(txt1):
    key = []
    keywords = rake.apply(txt1)
    for j in keywords:
        key.append(j[0])
    return key


def prefinalize(sent_l):
    sent_f = []
    for k in sent_l:
        k = remove_others(k)
        s = sent_analyze(k)
        if (s != None):
            sent_f.append(s)
    return sent_f


def finalize(sent_f):
    sent_final = []
    for g in sent_f:
        g = remove_stopwords(g)
        sent_final.append(g)
    sent_f = sent_final
    return sent_f

# Document Extraction
# -----------------------------------------------------------------
# keywords


def ka(sent_f):
    kw = []
    for z in sent_f:
        kw.append(keywords(z))
    return kw


def full(na, kw):
    full = []
    f = []
    for x in range(0, len(na)):
        f = na[x]+kw[x]
        full.append(f)
    return full
# nouns approach


def na(sent_f):
    na = []
    for x in sent_f:
        x1 = spacy_implement(x)
        na.append(x1)
    return na


def singlet(l):
    l1 = []
    for i in l:
        if (i not in l1):
            l1.append(i)
    return l1
# testinghaar


def postag(sent):
    text = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(text)
    return pos


def hello_world():

    df = pd.read_csv("test_data2.csv")
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 4'])
    # model 1 implementation on training data
    b = []
    for i_final in df['value']:
        sent_l = sent_seg(i_final)
        sent_f = prefinalize(sent_l)
        sent_f = finalize(sent_f)
        nab = na(sent_f)
        b.append(nab)
    return render_template('index.html', val=b)


if __name__ == "__main__":
    app.run(debug=True)
