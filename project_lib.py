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
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import warnings
warnings.filterwarnings('ignore')

from string import punctuation 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import urllib

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import sqlite3

''' twitterscraper allows to retrieve historical twitter data without using the
    Twitter API thus no need to have a premium account '''
    
''' begindate starts at date '''
''' enddate ends at date-1 '''


def get_text_from_image(url):
    local_filename, headers = urllib.request.urlretrieve(url)
    image_text = pytesseract.image_to_string(local_filename)
    print('Text: ' + image_text + ' retrieve from image.')
    return image_text

def retrieve_attachement_data(tweet):
    for match in re.finditer(r'((www\.[^\s]+)|(https?://[^\s]+))', tweet):
        url = tweet[match.start():match.end()]
        # use try in order to catch errors due to bad links
        try:
            # Retrieve link info
            #print('Testing url: ' + url)
            u = urllib.request.urlopen(url)
            result = u.info()
            
            # if the format is the one desired process it else skip it
            content_type = result.get_content_type()
            if len(content_type) < 5:
                print('When retrieving url ' + url + ' found content type ' + result + " that will not be analysed.")
            
            elif content_type[:5] == 'image': # match image/gif,image/jpeg,image/png
                tweet = tweet + ' ' + get_text_from_image(url)
            
            #elif content_type == ('application/pdf'):
                #tweet = tweet + ' ' + get_text_from_pdf(url)
                
        except urllib.error.HTTPError as e:
            print('An HTTPError has been raised, code ' + str(e.code) + '. Passing to the next step.')
        except urllib.error.URLError as e:
            print('An URLError has been raised, reason ' + str(e.reason) + '. Passing to the next step.')
        except ConnectionResetError:
            print('A ConnectionResetError has been raised. Passing to the next step.')

    return tweet

def download_attachement(tweet):
    
    return tweet

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


def multiprocess_execution(timestamp,img_urls,links,replies,retweets,text,tweet_id,comp):
    try:
        '''Initialize Sentiment Analyzer'''
        sid = SentimentIntensityAnalyzer()
        tweet_text = text
        if type(img_urls) is str:
            tweet_text = str(img_urls).replace('[\'','').replace('\']',' ') + ' ' + tweet_text
        if type(links) is str:
            tweet_text = str(links).replace('[\'','').replace('\']',' ') + ' ' + tweet_text
        #tweet_text = retrieve_attachement_data(tweet_text)
        tweet_text = tweet_normalized(tweet_text)
        polarity = sid.polarity_scores(tweet_text)
        returned_data = {'tweet_id':tweet_id, 'timestamp':timestamp, 'replies':replies, 'retweets':retweets,'clean_tweet':tweet_text,'Comp':polarity['compound'],'Negative':polarity['neg'],'Neutral':polarity['neu'],'Positive':polarity['pos']}
    except:
        print('An error ocurred, skipping the line ' + str(tweet_id) + ' on timestamp ' + str(timestamp))
        returned_data = {}
    return returned_data