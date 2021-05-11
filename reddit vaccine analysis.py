import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
nltk.download('wordnet')

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

rvm_df = pd.read_csv('/content/reddit_vm.csv')

nltk.download([
    "names",
    "stopwords",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

#convert body text to string to remove NaNs and make 1 dtype

rvm_df['body'] = rvm_df['body'].astype(str)

# collect all strings in body text into one string

body_all = ''
for text in rvm_df['body']:
  if text != 'nan':
      body_all += ' ' + text

# collect all strings in title text into one string

title_all = ''
for text in rvm_df['title']:
  if text != 'nan' and text != 'Comment':
      title_all += ' ' + text

#remove punctuation using regex

body_all = re.sub(r'[^\w\s]', '', body_all)
title_all = re.sub(r'[^\w\s]', '', title_all)

# tokenize all body text and convert to NLTK text type
body_text = nltk.Text(nltk.word_tokenize(body_all))

rvm_df['tokenized_text'] = rvm_df['body'].apply(nltk.word_tokenize)

def df_lemmatize(list):
  list = [wnl.lemmatize(word) for word in list]
  return list
  
  
def df_stopwords(list):
  list = [word for word in list if not word in stop_words]
  return list

rvm_df['stopped_text'] = rvm_df['tokenized_text'].apply(df_lemmatize)

rvm_df['lem_text'] = rvm_df['stopped_text'].apply(df_stopwords)

comments_cleaned = []
for text in rvm_df['lem_text']:
  for word in text:
    if word.isalpha() and word != 'nan':
      comments_cleaned.append(word)

fd_comments = nltk.FreqDist([w.lower() for w in comments_cleaned])

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

rvm_df['sentiment dict'] = rvm_df['body'].apply(sia.polarity_scores)

def sentmax(dict):
  maximum = max(dict.values())
  return maximum

rvm_df['sentmax'] = rvm_df['sentiment dict'].apply(sentmax)
