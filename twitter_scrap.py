#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:51:33 2020

@author: bernardo
"""

from twitterscraper import query_tweets
import datetime as dt 
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns

from textblob import TextBlob
import re
import string
import unicodedata
import contractions
import inflect

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


from string import punctuation 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

''' twitterscraper allows to retrieve historical twitter data without using the
    Twitter API thus no need to have a premium account '''
    
''' begindate starts at date '''
''' enddate ends at date-1 '''

def request_twitter(company, start_date, end_date):
    query = company + ' -filter:retweets -filter:replies'
    delta = dt.timedelta(days=7)
    twts_raw = pd.DataFrame()
    twts_df = pd.DataFrame()
    EndDT = start_date + delta
    BeginDT = start_date
    while EndDT <= EndDate:
    
        tweets = query_tweets(query, limit=None, begindate=BeginDT,
                              enddate=EndDT, lang='en')
        tweets_df = pd.DataFrame(t.__dict__ for t in tweets)
        twts_df = twts_df.append(tweets_df)
    
    
        if (EndDate - EndDT) > delta:
            BeginDT = BeginDT + delta
            EndDT = EndDT + delta
        else:
            BeginDT = BeginDT + delta
            EndDT = EndDate
            if BeginDT > EndDT:
                break
    twts_raw = twts_df        
    twts_df = twts_df.drop_duplicates(subset=['timestamp'])
    twts_df = twts_df.set_index('timestamp')
    twts_df = twts_df.sort_index()
    
    return twts_df, twts_raw

def remove_url(tweet):
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    return tweet

def remove_user(tweet):
    tweet = re.sub('@[^\s]+', '', tweet)
    return tweet

def remove_hashtags(tweet):
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

def contract_words(tweet):
    tweet = contractions.fix(tweet)
    return tweet

def remove_non_ascii(tweet):
    tweet = unicodedata.normalize('NFKD', tweet).encode('ascii', 'ignore').\
    decode('utf-8', 'ignore')
    return tweet
                                

def remove_characters(tweet):
    tweet = re.sub(r'[^a-zA-Z0-9]', ' ', tweet)
    return tweet

def lowercase(tweet):
    tweet = tweet.lower()
    return tweet

def remove_punctuation(tweet):
    tweet =re.sub(r'[^\w\s]', '', tweet)
    return tweet

def replace_numbers(tweet):
    p = inflect.engine()
    new_words = []
    for word in tweet.split():
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def remove_stopwords(tweet):
    clean_mess = [word for word in tweet.split() if word.lower() not in stopwords.words('english')]
    clean_tweet = ' '.join(clean_mess)
    return clean_tweet

def normalization(tweet):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet.split():
        normalized_text = lem.lemmatize(word, pos='v')
        normalized_tweet.append(normalized_text)
    return ' '.join(normalized_tweet)

def tweet_normalized(tweet):
    tweet = remove_url(tweet)
    tweet = remove_user(tweet)
    tweet = remove_hashtags(tweet)
    tweet = contract_words(tweet)
    tweet = remove_non_ascii(tweet)
    tweet = remove_characters(tweet)
    tweet = lowercase(tweet)
    tweet = remove_punctuation(tweet)
    tweet = replace_numbers(tweet)
    tweet = remove_stopwords(tweet)
    tweet = normalization(tweet)
    return tweet


#%%

''' REQUEST TWITTER DATA ''' 

'''LOOP TO REQUEST TWITTER DATA OF "NAMES" IN LIST companies. LOOP WILL SAVE DATA 
    IN 2 CSV FILES: ONE CSV WITH RAW DATA AND THE OTHER: SORTED, REMOVING 
    DUPLICATES, AND DATA IN FIRST COLUMN'''
    
###############################################################################
###############################################################################
''' data is taken as of BeginDate until (EndDate - 1) '''
BeginDate = dt.date(2017,1,1) 
EndDate = dt.date(2020,4,1)
companies = ['Microsoft', 'Apple', 'Mastercard','Intel Corp', 
             'Cisco Systems', 'Adobe', 'Nvidia',
             'Salesforce.com, Inc', 'PayPal', 'Oracle', '#SP500']
###############################################################################
###############################################################################
             
start_year = BeginDate.strftime('%Y%m%d')
end_year = EndDate.strftime('%Y%m%d')

for company in companies:
    temp_df = pd.DataFrame()
    temp_raw = []
    temp_df, temp_raw = request_twitter(company, BeginDate, EndDate)
    filename_df = company+'_'+start_year+'_'+end_year+'.csv'
    filename_raw = company+'_raw_'+start_year+'_'+end_year+'.csv'
    temp_df = temp_df.replace(r'\r',' ', regex=True)
    temp_raw = temp_raw.replace(r'\r',' ', regex=True)
    temp_df.to_csv(filename_df, sep = "|", header=True, index=True)
    temp_raw.to_csv(filename_raw, sep = "|",header=True,index=False)
    
#%%
''' CLEAN TWITTER DATA '''

''' - REMOVE SPECIAL CHARACTERS
    - REMOVE URL
    - REMOVE STOPWORDS
    - NORMALIZE WORDS: lower case, verbs, numbers to letter '''

data = pd.read_csv('Apple_20170101_20200401.csv', sep = "|")
#data = data.dropna(subset=['timestamp_epochs'])


''' tweets : data with only datetime and tweet text columns '''
tweets = data[['timestamp','text']]
''' tweets_clean: new dataframe that will store cleaned tweets '''
tweets_clean = pd.DataFrame(columns=['Date','Tweets'])
''' loop to clean text data 
    remove_characters removes special characters and url
    remove_stopwords removes stopwords '''
    
indx = 0
for indx, row in tqdm(tweets.iterrows()):
    tweet_text = row["text"]
    tweet_text = tweet_normalized(tweet_text)
    tweets_clean.sort_index()
    tweets_clean.at[indx, 'Date']= row["timestamp"]
    tweets_clean.at[indx, 'Tweets'] = tweet_text
    
''' save clean data to csv '''
tweets_clean.to_csv("Apple_clean_20170101_20200401.csv",sep='|', header=True, index=True)

#%%
''' SENTIMENT ANALYSIS: UNSUPERVISED ML ALGORITHM TO CLASSIFY
    EACH TWEET IN A SENTIMENT POSITIVE, NEUTRAL OR NEGATIVE

