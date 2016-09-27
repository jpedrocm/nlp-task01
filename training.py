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


def get_words_features(docs):
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
	fil = filter(lambda (k,v):avg<=v and (avg+std)>=v, freq_items)
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

	words_features = get_words_features(reuters.fileids())

	print len(words_features)
	#words= []
	#for doc in reuters.fileids("aql"):
	#	words.extend(reuters.words(doc))	

#	features = extract_feature(words,words_features)
#	od = sorted(features.items(), key=lambda (k,v): (v,k) , reverse = True)#sorted(features.items(),key=operator.itemgetter(0))
	
#	list_items = map(lambda (k,v): v , od)
#	avg = mean(list_items)
#	std = std_dev(list_items, avg)

#	for i in range(100):
#		print od[i]

	#print max(list_items)
	#print avg
	#print std
	"""for key,value in od:	
		print str(key)+" - "+str(value)
	"""
"""
	for doc in reuters.fileids():
		if doc.startswith('test'):
			test_set.append(doc)
		else:
			training_set.append(doc)

	print(len(training_set))
	print(len(test_set))
	#cat = reuters.categories();
	
	doc_acq = reuters.fileids("acq");
	doc = reuters.words(doc_acq[0]);
	words = list(set(doc))
	print(len(words))
	print(len(doc))
	freq = {}
	for w in words:
		freq[w] = doc.count(w)
	print freq
	#print(doc_acq)
	#print(cat)
	#print(len(cat))
"""
main()	