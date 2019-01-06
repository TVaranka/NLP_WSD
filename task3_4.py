import nltk
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import re
from nltk.wsd import lesk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score, f1_score


SV_SENSE_MAP = {
    "HARD1": ["difficult.a.01"],    # not easy, requiring great physical or mental
    "HARD2": ["hard.a.02"],          # dispassionate
    "HARD3": ["hard.a.03"],         # resisting weight or pressure
    "interest_1": ["interest.n.01"], # readiness to give attention
    "interest_2": ["interest.n.03"], # quality of causing attention to be given to
    "interest_3": ["pastime.n.01"],  # activity, etc. that one gives attention to
    "interest_4": ["sake.n.01"],     # advantage, advancement or favor
    "interest_5": ["interest.n.05"], # a share in a company or business
    "interest_6": ["interest.n.04"], # money paid for the use of money
    "cord": ["line.n.18"],          # something (as a cord or rope) that is long and thin and flexible
    "formation": ["line.n.01"], # a formation of people or things one beside another
    "text": ["line.n.05"],                 # text consisting of a row of words written across a page or computer screen
    "phone": ["telephone_line.n.02"],   # a telephone connection
    "product": ["line.n.22"],       # a particular kind of product or merchandise
    "division": ["line.n.29"],      # a conceptual separation or distinction
    "SERVE12": ["serve.v.02"],       # do duty or hold offices; serve in a specific function
    "SERVE10": ["serve.v.06"], # provide (usually but not necessarily food)
    "SERVE2": ["serve.v.01"],       # serve a purpose, role, or function
    "SERVE6": ["service.v.01"]      # be used by; as of a utility
}
#https://stackoverflow.com/questions/16381218/how-do-i-get-the-definition-for-a-sense-in-nltks-senseval-module



#TASK3
def nltk_lesk(target_word, stem=True, restrict_senses=True):
	"""
	3.Use the Lesk implementation in NLTK   (>>from nltk.wsd import lesk), test for few examples 
	of target words of seneval2 the result you will be getting using Lesk algorithm’s disambiguation 
	and report the result of the disambiguation task on both training and testing dataset. 
	Compare the result with Bayes’ classifier.
	"""
	#Store senseval objects
	instances = senseval.instances(target_word + ".pos")
	#Collect the raw sentences and ignore the pos tags
	raw = [[i[0] for i in instance.context] for instance in instances]
	y = np.array([instance.senses[0] for instance in instances])
	senses_unique = pd.Series(y).unique().tolist()

	if stem:
		porter_stemmer = PorterStemmer()
		raw = [[porter_stemmer.stem(w) for w in sent] for sent in raw]

	if restrict_senses:
		X = np.array([lesk(sent, target_word, synsets=[wn.synset(SV_SENSE_MAP[sense][0]) for sense in senses_unique]
							).name() for sent in raw])
	else:
		X = np.array([lesk(sent, target_word).name() for sent in raw])
	

	#Code the nltk senses to match senseval's senses
	[np.place(X, X == SV_SENSE_MAP[sense], sense) for sense in senses_unique]


	accuracy_lesk = (X == y).mean()
	print("Lesk accuracy: {}".format(accuracy_lesk))
	print("Precision {}\nRecall {}".format(precision_score(y, X, average="macro"),
		recall_score(y, X, average="macro")))
	return [accuracy_lesk, precision_score(y, X, average="macro"), recall_score(y, X, average="macro"),
			f1_score(y, X, average="macro")]



#TASK4
def improved_lesk(target_word, stem=True, remove_stop_words=True, remove_punctuation=True, normalize=True):
	"""
	4.We want to expand the Lesk algorithm to include related terms. Start with NLTK 
	WordNet examples to design and implement a program that extract for each sense of the 
	target word, all the information in the synset (word itself, sense and example sentence (s)). 
	Now expand for each term of the example sentence its direct hyponym, direct hypernym, antonym 
	(if available) and any other related terms that can extracted using WordNet entity relationship. 
	Now construct a large set of each sense of target word that includes all the above. Then perform 
	the simplified Lesk algorithm disambiguation methodology, e.g., count the number of common words 
	between the created enlarged list of each sense and the set of words in the sentence to be disambiguated.
	"""
	#Store senseval objects
	instances = senseval.instances(target_word + ".pos")
	stop_words = set(stopwords.words('english'))

	def _get_name(sense):
		"""Using regular expressions extract the name from sense.
		E.g. 'difficult.a.01' -> 'difficult' """
		return re.match(r'[a-z]+', sense._name).group()

	def _create_features(target_word):
		features = {}
		for sense in wn.synsets(target_word):
			sense_name = _get_name(sense)
			features[sense._name] = []
			#features[sense._name].extend(sense.definition().split())
			
			
			for definition_word in sense.definition().split():
				for definition_sense in wn.synsets(definition_word):
					for definition_sense_hypernym in definition_sense.hypernyms():
						features[sense._name].append(_get_name(definition_sense_hypernym))
					for definition_sense_hyponym in definition_sense.hyponyms():
						features[sense._name].append(_get_name(definition_sense_hyponym))
					
					for definition_sense_member_holonym in definition_sense.member_holonyms():
						features[sense._name].append(_get_name(definition_sense_member_holonym))

					for definition_sense_member_meronym in definition_sense.member_meronyms():
						features[sense._name].append(_get_name(definition_sense_member_meronym))
					
					for definition_sense_part_holonym in definition_sense.part_meronyms():
						features[sense._name].append(_get_name(definition_sense_part_holonym))
					
			
			for example_sentence in sense.examples():
				features[sense._name].extend(example_sentence.split())
				for word in example_sentence.split():
					for example_sense in wn.synsets(word):
						
						for example_sense_hypernym in example_sense.hypernyms():
							features[sense._name].append(_get_name(example_sense_hypernym))
						for example_sense_hyponym in example_sense.hyponyms():
							features[sense._name].append(_get_name(example_sense_hyponym))
						for example_sense_member_holonym in example_sense.member_holonyms():
							features[sense._name].append(_get_name(example_sense_member_holonym))
						
						for example_sense_member_meronym in example_sense.member_meronyms():
							features[sense._name].append(_get_name(example_sense_member_meronym))
						
						for example_sense_part_holonym in example_sense.part_meronyms():
							features[sense._name].append(_get_name(example_sense_part_holonym))

			
			features[sense._name] = set(features[sense._name])

		return features		

	features = _create_features(target_word)
	y = np.array([instance.senses[0] for instance in instances])
	senses_unique = pd.Series(y).unique().tolist()	
	
	porter_stemmer = PorterStemmer()
	#Use only the features from the senses in the dataset
	features_of_possible_senses = [list(features[SV_SENSE_MAP[sense][0]])for sense in senses_unique]
	#Stem the feature words
	if stem:
		features_of_possible_senses = [[porter_stemmer.stem(w) for w in feature_set] for feature_set in features_of_possible_senses]

	if remove_stop_words:
		features_of_possible_senses = [[w for w in feature_set if w not in stop_words] for feature_set in features_of_possible_senses]


	def sense_for_sentence(sent):
		"""
		Calculate the intersection between the given sentence and the different senses
		And return the sense with the highest number of intersections
		"""
		
		intersections = np.array([len(set(feature_set).intersection(sent)) for feature_set in features_of_possible_senses])
		if normalize:
			intersections = intersections.flatten() / np.array([len(feature) for feature in features_of_possible_senses])
		argmax = np.argmax(intersections)
		return senses_unique[argmax]

	results = []
	for instance in instances:
		sent = [w[0] for w in instance.context]
		if remove_punctuation:
			sent = [re.sub(r'[^\w\s]', '', w) for w in sent]
		if stem:
			sent = [porter_stemmer.stem(w[0]) for w in instance.context]
		if remove_stop_words:
			sent = [w for w in sent if w not in stop_words]
		results.append(sense_for_sentence(sent))

	X = np.array(results)
	

	accuracy_improved_lesk = (X == y).mean()
	print(accuracy_improved_lesk)

	from sklearn.metrics import confusion_matrix
	print(confusion_matrix(X, y))

	print("Precision {}\nRecall {}".format(precision_score(y, X, average="macro"),
		recall_score(y, X, average="macro")))
	return [accuracy_improved_lesk, precision_score(y, X, average="macro"), recall_score(y, X, average="macro"),
			f1_score(y, X, average="macro")]

#nltk_lesk("serve", False, True)
#improved_lesk("hard", True, True)

#"""
final_results = []
for word in ["hard", "interest", "line", "serve"]:
	final_results.append(improved_lesk(word, True, False, True, True))

#improved_lesk returns
#0 accuracy
#1 precision
#2 recall
#3 f1-score
final_results = np.array(final_results)
print(final_results[:, 3].mean())
#"""

"""
F1-scores
			stemming - remove stop words - remove punctuation - normalize - |all combined
HARD                  		                                                |
interest
line
serve
-----------------------------------------------------------------------------------------
all
"""