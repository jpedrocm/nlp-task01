import math
import operator
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import reuters

def mean(list_items):
    return sum(list_items)/len(list_items)

def std_dev(list_items, mean_items):
    variance_list = map(lambda x : pow(x-mean_items, 2), list_items)
    return math.sqrt(sum(variance_list)/len(list_items))


def get_words_features():
	list_words = []
	for doc in reuters.fileids():
		words = reuters.words(doc)
		list_words.extend(words)

	words_features = list(set(list_words))
	freq = extract_feature(list_words,words_features)
	freq_items = freq.items()
	list_freq = map(lambda (k,v): v , freq_items)
	avg = mean(list_freq)
	std = std_dev(list_freq, avg)
	fil = filter(lambda (k,v):avg-std/100<=v and (avg+std/100)>=v, freq_items)
	return fil

def extract_feature(words, features):
	freq = {}
	for w in words:
		if(w in freq):
			freq[w] = freq[w] + 1
		else:
			freq[w] = 1	
	return freq


def main():
	training_set = []
	test_set = []

	bag_of_words = get_words_features()

	classes = ["acq","earn","money-fx","grain","crude","trade","interest","ship","wheat","corn"]

	#print classes

	#print reuters.categories();

	train_set = []
	test_set = []

	for doc in reuters.fileids("acq"):
		words = reuters.words(doc)
		feat = extract_feature(words,bag_of_words)
		inst = (feat,"acq")
		if doc.startswith("train"):
			train_set.append(inst)
		else:
			test_set.append(inst)	

	non_docs = set(reuters.fileids(classes)) - set(reuters.fileids("acq"));

	for doc in non_docs:
		words = reuters.words(doc)
		feat = extract_feature(words,bag_of_words)
		inst = (feat,"non_acq")
		if doc.startswith("train"):
			train_set.append(inst)
		else:
			test_set.append(inst)

	print len(train_set)	

main()	