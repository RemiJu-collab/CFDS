#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:10:29 2020

@author: bernardo
"""

import pandas as pd
import numpy as np
import csv
import re

from textblob import TextBlob
import string
import preprocessor as p

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import matplotlib.pyplot as plt

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

''' read saved tweets per company '''

companies = ['Microsoft', 'Apple', 'Mastercard', 'Intel', 
             'Cisco_Systems', 'GDP', 'Dow_Jones']
path = os.path.abspath(os.getcwd())+"/"          
file_end = "_clean_20170101_20200401.csv"

companylist = {}
for company in companies:
    filename = path+company+file_end
    companylist[company] =  pd.read_csv(filename,sep='|',index_col = 0)

''' Sentiment Analysis '''
# Sentiment for each Tweet ----------------------------------------------------

sentiment_pt = SentimentIntensityAnalyzer()
companylist_sentiment = {}
for company, df in companylist.items():
    df["Comp"] = ''
    df["Negative"] = ''
    df["Neutral"] = ''
    df["Positive"] = ''
    df = df.dropna(subset=['Tweets']).reset_index(drop=True)
    for indexx, row in tqdm(df.T.iteritems()):
        try:            
            sentence_pt=unicodedata.normalize('NFKD', df.loc[indexx, 'Tweets'])
            sentence_pt_sentiment=sentiment_pt.polarity_scores(sentence_pt)
            df.at[indexx, 'Comp'] = sentence_pt_sentiment['compound']
            df.at[indexx, 'Negative'] = sentence_pt_sentiment['neg']
            df.at[indexx, 'Neutral'] = sentence_pt_sentiment['neu']
            df.at[indexx, 'Positive'] = sentence_pt_sentiment['pos']
        except TypeError:
            print(indexx)
    companylist_sentiment[company] = df

''' save to csv with sentiment scores '''
for company, df in companylist_sentiment.items():
    filename = company+"_sentiment_tweets.csv"
    df.to_csv(filename,sep='|',header=True, index=False)


''' one sentiment per day, takes the average of the sentiment scores '''

sentiment_per_date = {}

for company,df in companylist_sentiment.items():
    feat = pd.concat([df.Date, df.Comp, df.Positive,df.Negative],axis = 1)
    feat.Comp = pd.to_numeric(feat.Comp)
    feat.Positive = pd.to_numeric(feat.Positive)
    feat.Negative = pd.to_numeric(feat.Negative)
    feat.Date = pd.to_datetime(feat.Date).dt.date
    feat_agg = feat.groupby('Date').mean()
    sentiment_per_date[company] = feat_agg

''' save to csv with sentiment scores '''
for company, df in sentiment_per_date.items():
    filename = company+"_daily_sentiment_tweets.csv"
    df.to_csv(filename,sep='|',header=True, index=True)
