# Library of functions used in notebooks for scraping and analysing the website of
# SEG Geophysics journal (http://library.seg.org/loi/gpysa7)
#
# 19-11-2017
# M. Ravasi

import os
import glob 
from datetime import datetime,date
import random
import itertools
import pickle
import time

import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import names
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

from IPython.display import display



def get_date(date_text):
	""" 
	Extract date from text in format yyyy-mm-dd 00:00:00 
	"""
	date_text = date_text.split(':')[1][1:].split()
	#print date_text[0]+' '+date_text[1][:3]+' '+date_text[2]
	if len(date_text)<3:
		date_text = date(1900, 7, 14)
	else:    
		date_text = datetime.strptime(date_text[0]+' '+date_text[1][:3]+' '+date_text[2], '%d %b %Y')
	return date_text


def get_pubhistory(dates):
	""" 
	Extract publication history from list of dates in text format yyyy-mm-dd 00:00:00 
	"""
	# create publication history
	if len(dates)==0:
		received  = date(1900, 7, 14)
		accepted = published = received
	elif len(dates)==1:
		received  = get_date(dates[0])
		accepted = published = received
	elif len(dates)==2:
		received  = get_date(dates[0])
		accepted  = get_date(dates[1])
		published = accepted
	else:
		received  = get_date(dates[0])
		accepted  = get_date(dates[1])
		published = get_date(dates[2])
	
	return received, accepted, published 


def find_categories(html):
	""" 
	Find number of papers for each category in the html of a journal issue
	"""
	# read html issue page and extract categories for each paper
	soup = BeautifulSoup(html, "html5lib")
	infos = soup.findAll('div', { "class" : "subject" })

	# remove parenthesis from categories to be able to do regex
	infos_reg = [str(info).replace('(','.*?') for info in infos]
	infos_reg = [info.replace(')','.*?') for info in infos_reg]
	#print infos

	categories=[]
	for iinfo in range(len(infos)-1): 
		infostr = '(('+str(infos_reg[iinfo])+').*?('+str(infos_reg[iinfo+1])+'))'
		#print infostr
		dois = re.findall(unicode(infostr, "utf-8"), html, re.DOTALL) 
		#print dois[0][0]
		dois = re.findall('"((/doi/abs).*?)"', dois[0][0])
		#print dois

		#category = re.findall(r'subject">(.*)</div>', str(infos[iinfo]))
		category = re.findall(r'subject">(.*)</div>', str(infos[iinfo]))[0].decode("utf-8")
		print '%s: %d' %(category, len(dois)/2)

		categories.extend([category]*(len(dois)/2))

	return categories


def words_from_text(texts):
	""" 
    Loop through list of strings and extract all words and remove common stopwords
    """ 
	words = []
	# extract words and make them lower case
	for text in texts:
		tokens = re.findall('\w+', text)
		for word in tokens:
			words.append(word.lower())

	# get English stopwords and remove them from list of words
	sw = nltk.corpus.stopwords.words('english')

	# add sklearn stopwords to words_sw
	sw = set(sw + list(ENGLISH_STOP_WORDS))

	# add to words_ns all words that are in words but not in sw
	words_ns = []
	for word in words:
		if word not in sw:
			words_ns.append(word)
	#print words_ns
	return words_ns
    

def extract_first_authors_name(author_list):
	""" 
    Extract first name from a string including list of authors in form Author1; Author2; ...; AuthorN
    """ 
	return author_list.split(';')[0].split(' ')[0]


def gender_features(word):
    """ 
    Feature extractor for the name classifier
    
    The feature evaluated here is the last letter of a name
    feature name - "last_letter"
    """
    return {"last_letter": word[-1]}  # feature set


def gender_training(verb=False):
    """ 
    Gender training based on nltk.NaiveBayesClassifier
    """
    # Extract the data sets
    labeled_names = ([(name, "male") for name in names.words("male.txt")] +
                     [(name, "female") for name in names.words("female.txt")])

    # Shuffle the names in the list
    random.shuffle(labeled_names)

    # Process the names through feature extractor
    feature_sets = [(gender_features(n), gender)
                    for (n, gender) in labeled_names]

    # Divide the feature sets into training and test sets
    train_set, test_set = feature_sets[500:], feature_sets[:500]

    # Train the naiveBayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    if verb:
        # Test the accuracy of the classifier on the test data
        print('Accuracy: %f ' % nltk.classify.accuracy(classifier, test_set))
        print classifier.show_most_informative_features(5)
        
    return classifier


def gender_classifier(name, classifier):
    """ 
    Apply gender classifier to a name
    """
    return classifier.classify(gender_features(name))


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
