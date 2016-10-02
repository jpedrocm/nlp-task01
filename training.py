import math
import operator
import collections
import nltk
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.metrics import *
from nltk.classify import *

CLASSES = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
STOPWORDS = stopwords.words('english')
PUNCTUATION_LIST = list(string.punctuation)

def mean(list_items):
    return sum(list_items)/float(len(list_items))

def std_dev(list_items, mean_items):
    variance_list = map(lambda x : pow(x-mean_items, 2), list_items)
    return math.sqrt(sum(variance_list)/float(len(list_items)))

def f_measure(precision, recall):
	return (2*precision*recall) / float(precision+recall)

def precision(tp, fp):
	return tp / float(tp+fp)

def accuracy(tp, tn, fp, fn):
	numerator = tp+tn
	return float(numerator) / float(numerator+fp+fn)

def recall(tp, fn):
	return tp / float(tp+fn)

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

	prec = precision(tp,fp)
	rec = recall(tp,fn)
	acc = accuracy(tp, tn, fp, fn)
	f1 = f_measure(prec, rec)

	return {'category': cat.upper(), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': prec, 'accuracy': acc, 'recall': rec, 'fmeasure': f1}

def print_metrics(metrics, macro = False):
	print metrics['category']
	if not macro:
		print "True Positives: " + str(metrics['tp'])
		print "True Negatives: " + str(metrics['tn'])
		print "False Positives: " + str(metrics['fp'])
		print "False Negatives: " + str(metrics['fn'])
	print "Accuracy: " + str(metrics['accuracy'])
	print "Precision: " + str(metrics['precision'])
	print "Recall: " + str(metrics['recall'])
	print "F-measure: " + str(metrics['fmeasure'])
	print ""

def calculate_micro_metrics(list_of_metrics, model):
	total_tp = calculate_micro_metric(list_of_metrics, 'tp')
	total_tn = calculate_micro_metric(list_of_metrics, 'tn')
	total_fp = calculate_micro_metric(list_of_metrics, 'fp')
	total_fn = calculate_micro_metric(list_of_metrics, 'fn')

	prec = precision(total_tp, total_fp)
	rec = recall(total_tp, total_fn)
	acc = accuracy(total_tp, total_tn, total_fp, total_fn)
	f1 = f_measure(prec, rec)

	return {'category': model.upper(), 'tp': total_tp, 'tn': total_tn, 'fp': total_fp, 'fn': total_fn,
	 'precision': prec, 'accuracy': acc, 'recall': rec, 'fmeasure': f1}

def calculate_micro_metric(list_of_metrics, metric_name):
	list_of_metric = map(lambda x: x[metric_name], list_of_metrics);
	return sum(list_of_metric)

def calculate_macro_metrics(list_of_metrics, model):
	mean_prec = calculate_macro_metric(list_of_metrics, 'precision')
	mean_rec = calculate_macro_metric(list_of_metrics, 'recall')
	mean_fmeasure = calculate_macro_metric(list_of_metrics, 'fmeasure')
	mean_acc = calculate_macro_metric(list_of_metrics, 'accuracy')

	return {'precision': mean_prec, 'recall': mean_rec, 'fmeasure': mean_fmeasure, 'accuracy': mean_acc, 'category': model.upper()}

def calculate_macro_metric(list_of_metrics, metric_name):
	list_of_metric = map(lambda x: x[metric_name], list_of_metrics)
	return mean(list_of_metric)

def get_words_features():
	list_words = []

	for category in CLASSES:
		category_words = []
		for doc in reuters.fileids(category):
			if doc.startswith('train'):
				words = reuters.words(doc)
				words = map(lambda w: w.lower(), words)
				category_words.extend(words)
		category_words = remove_stopwords_and_punctuation(category_words)
		category_words = filter_words_category(category_words)
		list_words.extend(category_words)

	return list_words

def remove_stopwords_and_punctuation(words):
	return filter(lambda x: x not in STOPWORDS and x not in PUNCTUATION_LIST, words)

def filter_words_category(words):
	words_features = list(set(words))
	freq = count_frequency(words)
	freq_items = freq.items()
	list_freq = map(lambda (k,v): v , freq_items)
	avg = mean(list_freq)
	std = std_dev(list_freq, avg)
	fil = filter(lambda (k,v): (avg-std/100) <=v and (avg+std/100)>=v, freq_items)

	return map(lambda (k,v): k, fil)

def count_frequency(words):
	freq = {}
	for w in words:
		if(w in freq):
			freq[w] += 1
		else:
			freq[w] = 1	

	return freq

def extract_feature(words, features):
	freq = {}
	for w in words:
		if w in features:
			if(w in freq):
				freq[w] += 1
			else:
				freq[w] = 1
	return freq

def get_sets(category_name, bag_of_words, docs):
	train_set = []
	test_set = []

	for doc in docs:
		words_in_doc = map(lambda w: w.lower(), reuters.words(doc))
		words_in_doc = remove_stopwords_and_punctuation(words_in_doc)
		feat = extract_feature(words_in_doc, bag_of_words)
		inst = (feat, category_name)
		if doc.startswith("train"):
			train_set.append(inst)
		else:
			test_set.append(inst)

	return (train_set, test_set)

def get_sets_from_category(category, bag_of_words):

	docs = reuters.fileids(category)
	(train_set, test_set) = get_sets(category, bag_of_words, docs)

	non_docs = set(reuters.fileids(CLASSES)) - set(reuters.fileids(category));
	(train_set_remaining, test_set_remaining) = get_sets("non_"+category, bag_of_words, non_docs)

	train_set.extend(train_set_remaining)
	test_set.extend(test_set_remaining)
	
	return (train_set, test_set)

def main():
	bag_of_words = get_words_features()
	metrics_per_category_classifier = []

	for cat in CLASSES:
		train_set, test_set = get_sets_from_category(cat, bag_of_words)
		naive_classifier = nltk.NaiveBayesClassifier.train(train_set)
		ref = []
		resu = []
		for (feat,label) in test_set:
			ref.append(label)
			observed = naive_classifier.classify(feat)
			resu.append(observed)

		category_classifier_metrics = get_metrics(ref, resu, cat)
		print_metrics(category_classifier_metrics)
		metrics_per_category_classifier.append(category_classifier_metrics)

	macro_metrics = calculate_macro_metrics(metrics_per_category_classifier, 'Naive Bayes Classifier')
	print "Macro Metrics"
	print_metrics(macro_metrics, macro = True)

	micro_metrics = calculate_micro_metrics(metrics_per_category_classifier, 'Naive Bayes Classifier')
	print "Micro Metrics"
	print_metrics(micro_metrics)
			
main()