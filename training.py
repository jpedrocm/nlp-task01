import math
import operator
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import reuters
from nltk.metrics import *
from nltk.classify import *

CATEGORIES = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

def mean(list_items):
    return sum(list_items)/len(list_items)

def std_dev(list_items, mean_items):
    variance_list = map(lambda x : pow(x-mean_items, 2), list_items)
    return math.sqrt(sum(variance_list)/len(list_items))

def f_measure(precision, recall):
	return (2*precision*recall) / float(precision+recall)

def precision(tp, fp):
	return tp / float(tp+fp)

def accuracy(tp, tn, fp, fn):
	numerator = tp+tn
	return float(numerator) / float(numerator+fp+fn)

def recall(tp, fn):
	return tp / float(tp+fn)

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

def get_sets_from_category(category, bag_of_words):
	train_set = []
	test_set = []

	for doc in reuters.fileids(category):
		words = reuters.words(doc)
		feat = extract_feature(words,bag_of_words)
		inst = (feat,category)
		if doc.startswith("train"):
			train_set.append(inst)
		else:
			test_set.append(inst)	

	non_docs = set(reuters.fileids(CLASSES)) - set(reuters.fileids(category));

	for doc in non_docs:
		words = reuters.words(doc)
		feat = extract_feature(words,bag_of_words)
		inst = (feat,"non_"+category)
		if doc.startswith("train"):
			train_set.append(inst)
		else:
			test_set.append(inst)
	
	return (train_set,test_set)

def updateValuesToMicroMean(tp, tn, fp, fn):

def get_metrics(ref, resu, cat):
	ref_set = set(ref)
	resu_set = set(resu)
	
	clabel = cat;
	non_clabel = "non_"+cat;

	conf_matrix = ConfusionMatrix(ref, resu)
	tp = conf_matrix[clabel,clabel]
	tn = conf_matrix[non_clabel,non_clabel]
	fn = conf_matrix[clabel,non_clabel]
	fp = conf_matrix[non_clabel,clabel]

	updateValuesToMicroMean(tp, tn, fp, fn)

	print "tp "+str(tp)
	print "tn "+str(tn)
	print "fn "+str(fn)
	print "fp "+str(fp)

	prec = precision(tp,fp)
	rec = recall(tp,fn)

	print "Acuracy ("+cat+") "+str(accuracy(tp,tn,fp,fn))
	print "Precision ("+cat+") "+str(prec)
	print "Recall ("+cat+") "+str(rec)
	print "F-Measure ("+cat+") "+str(f_measure(prec,rec))
	print "--------------------------------------------------------------------------------"

def main():
	bag_of_words = get_words_features()

	for cat in CLASSES:
		train_set, test_set = get_sets_from_category(cat, bag_of_words)
		naive_classifier = nltk.NaiveBayesClassifier.train(train_set)
		ref = []
		resu = []
		for (feat,label) in test_set:
			ref.append(label)
			observed = naive_classifier.classify(feat)
			resu.append(observed)

		get_metrics(ref,resu,cat)
		

		maxent_classif = nltk.classify.DecisionTreeClassifier.train(train_set)

		ref = []
		resu = []
		for (feat,label) in test_set:
			ref.append(label)
			observed = maxent_classif.classify(feat)
			resu.append(observed)

		get_metrics(ref,resu,cat)
				
main()	