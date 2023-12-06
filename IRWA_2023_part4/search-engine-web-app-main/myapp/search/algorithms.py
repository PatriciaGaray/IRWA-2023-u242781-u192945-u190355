import json
import re
import string
import math 
import collections
from collections import Counter, defaultdict
from array import array

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import pickle
from numpy import linalg as la
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
import nltk
import demoji

from myapp.search.load_corpus import load_corpus_as_df

nltk.download('stopwords')

pkl_file_path = 'index_tfidf_data.pkl'
pkl_file_path2 = 'index_word2vec_data.pkl'
pkl_file_path3 = 'index_bm25_data.pkl'


# Load the data from the pickle files
with open(pkl_file_path, 'rb') as pkl_file:
    loaded_index_data = pickle.load(pkl_file)

with open(pkl_file_path2, 'rb') as pkl_file2:
    loaded_index_data2 = pickle.load(pkl_file2)

with open(pkl_file_path3, 'rb') as pkl_file3:
    loaded_index_data3 = pickle.load(pkl_file3)

def build_terms(line):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Remove URLs using a regular expression
    line = re.sub(r'http\S+|www\S+|https\S+', '', line)
    # Replace special characters
    line = line.replace('’', ' ').replace('“', ' ').replace('”', ' ').replace('‘', '')
    line = line.lower()  # Transform in lowercase
    line = demoji.replace(line, '')  # Remove emojis

    # Tokenize the text while preserving hashtags
    tokens = re.findall(r'\w+|#\w+', line)
    # Remove punctuation from words (excluding hashtags)
    tokens = [re.sub(r'[{}]'.format(string.punctuation), '', token) if not token.startswith('#') else token for token in tokens]
    tokens = [w for w in tokens if w not in stop_words]  # Eliminate stopwords
    tokens = [stemmer.stem(w) for w in tokens]  # Perform stemming

    return tokens

####### TF-IDF #######
def rank_documents(terms, tweets, index, idf, tf):
    # We are interested only on the element of the docVector corresponding to the query terms
    tweet_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue
        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm*idf[term]


        # Generate tweet_vectors for matching tweets
        for tweet_index, (tweet, postings) in enumerate(index[term]):
            if tweet in tweets:
              tweet_vectors[tweet][termIndex] = tf[term][tweet_index] * idf[term]

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    tweet_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in tweet_vectors.items() ]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]
    tweet_scores= [x[0] for x in tweet_scores]

    if len(result_tweets) == 0:
        print("No results found, try again")
        query = input()
        tweets = search_tf_idf(query, index, idf, tf)

    return result_tweets, tweet_scores

def search_tf_idf(query, index, idf, tf):
    """
    output is the list of tweets that contain any of the query terms.
    So, we will get the list of tweets for each query term, and take the union of them.
    """
    query = build_terms(query)
    tweets = set()
    for term in query:
        try:
            # store in term_tweets the ids of the tweets that contain "term"
            term_tweets=[posting[0] for posting in index[term]]

            # tweets = tweets Union term_tweets
            tweets = tweets.union(term_tweets)
        except:
            #term is not in index
            pass
    tweets = list(tweets)
    result_tweets, tweet_scores = rank_documents(query, tweets, index, idf, tf)
    return result_tweets, tweet_scores

####### Word2Vec #######
def rank_documents_word2vec(query_terms, tweets, word2vector, model):
    """
    Perform the ranking of the results of a search based on Word2Vec word vectors

    Arguments:
    query_terms -- list of query terms
    tweets -- list of tweets, to rank, matching the query
    word2vector -- word vectors for each term in each tweet

    Returns:
    Print the list of ranked documents
    """

    # Compute the query_vector
    query_vector = np.array([model.wv[term] for term in query_terms if term in model.wv.key_to_index])
    query_vector = np.mean(query_vector, axis=0)

    # Calculate the score of each doc
    tweet_scores = [[np.dot(tweet_vector, query_vector), tweet] for tweet, tweet_vector in word2vector.items() if tweet in tweets] #if tweet in tweets
    tweet_scores.sort(reverse=True)

    result_tweets = [x[1] for x in tweet_scores]
    tweet_scores = [x[0] for x in tweet_scores]

    if len(tweet_scores) == 0:
      print("No results found, try again")

    return result_tweets, tweet_scores

def search_word2vec(query, index, word2vector, model):
    """
    Output is the list of tweets that contain any of the query terms.
    So, we will get the list of tweets for each query term, and take the union of them.
    """
    query = build_terms(query)
    print(query)
    tweets = set()
    for term in query:
        try:
            # Store in term_tweets the ids of the tweets that contain "term"
            term_tweets = [tweet for tweet in index[term]]
            # Tweets = tweets Union term_tweets
            tweets = tweets.union(term_tweets)
        except:
            # Term is not in index
            pass

    tweets = list(tweets)
    print(tweets)
    ranked_tweets, tweet_scores = rank_documents_word2vec(query, tweets, word2vector, model)
    return ranked_tweets, tweet_scores

####### BM25 #######
def rank_documents_bm25_with_popularity(terms, tweets, index, idf, tf, avg_doc_length, tweets_df):
    tweet_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        # Compute BM25 weights for the query terms
        k1 = 1.5
        b = 0.75
        tf_query = query_terms_count[term]
        idf_term = idf[term]

        query_vector[termIndex] = ((k1 + 1) * tf_query) / (
                k1 * ((1 - b) + b * (avg_doc_length / avg_doc_length)) + tf_query) * idf_term

        # Generate tweet_vectors for matching tweets
        for tweet_index, (tweet, postings) in enumerate(index[term]):
            if tweet in tweets:
                # BM25 Calculus
                tf_tweet = tf[term][tweet_index]
                BM25_score = ((tf_tweet * (2.0 + 1.0)) / (
                        tf_tweet + 2.0 * (1.0 - b + b * (avg_doc_length / avg_doc_length))))
                tweet_vectors[tweet][termIndex] = BM25_score

    # Consider popularity metrics
    for tweet_id in tweet_vectors:
        likes = tweets_df.loc[tweets_df['id'] == tweet_id, 'likes'].values[0]
        retweets = tweets_df.loc[tweets_df['id'] == tweet_id, 'retweets'].values[0]

        # You can adjust the weights according to your preference
        popularity_score = 0.6 * likes + 0.4 * retweets

        # Combine BM25 score and popularity score
        tweet_vectors[tweet_id] = [0.8 * bm25_score + 0.2 * popularity_score for bm25_score in
                                    tweet_vectors[tweet_id]]

    # Calculate the score of each doc
    tweet_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in tweet_vectors.items()]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]
    tweet_scores = [x[0] for x in tweet_scores]

    if len(result_tweets) == 0:
        print("No results found, try again")
        query = input()
        tweets = search_bm25_with_popularity(query, index, idf, tf, avg_doc_length, tweets_df)

    return result_tweets, tweet_scores


def search_bm25_with_popularity(query, index, avg_doc_length, tweets_df, idf, tf):
    query = build_terms(query)
    tweets = set()
    for term in query:
        try:
            term_tweets = [posting[0] for posting in index[term]]
            tweets = tweets.union(term_tweets)
        except:
            pass
    tweets = list(tweets)
    ranked_tweets, tweet_scores = rank_documents_bm25_with_popularity(query, tweets, index, idf, tf, avg_doc_length,
                                                                      tweets_df)
    return ranked_tweets, tweet_scores



def load_preprocessed_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def search_in_corpus(query, corpus, search_option):

    # para no tener que preprocessar todo el dataset cada vez:
    tweets_df = pd.DataFrame(load_preprocessed_tweets('preprocessed_tweets.json'))
    tweets_df.rename(columns={'tweet_id': 'id'}, inplace=True)

    if (search_option == "TF-IDF"):

        # Usa los datos almacenados en index_data en lugar de recalcularlos
    
        index = loaded_index_data['index']
        tf = loaded_index_data['tf']
        df = loaded_index_data['df']
        idf = loaded_index_data['idf']
        tweet_id_index = loaded_index_data['tweet_id_index']

        # 2. apply ranking  
        ranked_tweets, result_tweets = search_tf_idf(query, index, idf, tf)
        return ranked_tweets
    
    elif (search_option == "Word2Vec"):

        tweets = tweets_df['tweet'].tolist()
        model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)

        # index 
        index = loaded_index_data2['index']
        word2vector = loaded_index_data2['word2vector']

        # Apply Word2Vec search
        ranked_tweets, result_tweets = search_word2vec(query, index, word2vector, model)
        return ranked_tweets
    
    elif (search_option == "BM25"):
        
        # index
        index = loaded_index_data3['index']
        tf = loaded_index_data3['tf']
        df = loaded_index_data3['df']
        idf = loaded_index_data3['idf']
        tweet_id_index = loaded_index_data3['tweet_id_index']
        avg_doc_length = loaded_index_data3['avg_doc_length']

        ranked_tweets, tweet_scores = search_bm25_with_popularity(query, index, avg_doc_length, tweets_df, idf, tf)
        return ranked_tweets

  
    return ''
