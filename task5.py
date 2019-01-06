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
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

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


glove2word2vec(glove_input_file="glove.6B/glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
#glove2word2vec(glove_input_file="glove.twitter.27B/glove.twitter.27B.200d.txt", word2vec_output_file="gensim_glove_vectors.txt")
#glove2word2vec(glove_input_file="glove.42B.300d/glove.42B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")

glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

def _get_name(sense):
	"""Using regular expressions extract the name from sense.
	E.g. 'difficult.a.01' -> 'difficult' """
	return re.match(r'[a-z]+', sense._name).group()


def _create_features(target_word):
	features = {}
	for sense in wn.synsets(target_word):
		sense_name = _get_name(sense)
		features[sense._name] = []
		features[sense._name].extend(sense.definition().split())


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


def _sense_for_sentence(sent, features_of_possible_senses, senses_unique):
	"""
	Calculate the cosine similarity between the given sentence and the different senses
	And return the sense with the highest number of intersections
	"""
	similarities = []
	for feature_set in features_of_possible_senses:
		similarity = []
		for w1 in feature_set:
			temp = np.array([glove_model.similarity(w1, w2) for w2 in sent])
			similarity.append(temp.mean())
			#temp = np.where(temp < 0.95, 0, 1)
			#similarity.append(temp.sum())
		similarities.append(np.mean(similarity))
	return senses_unique[np.argmax(similarities)]


def word_embeddings(target_word):
	instances = senseval.instances(target_word + ".pos")
	stop_words = set(stopwords.words('english'))

	features = _create_features(target_word)

	y = np.array([instance.senses[0] for instance in instances])
	senses_unique = pd.Series(y).unique().tolist()

	porter_stemmer = PorterStemmer()
	#Use only the features from the senses in the dataset
	features_of_possible_senses = [list(features[SV_SENSE_MAP[sense][0]])for sense in senses_unique]
	#Remove the words not in the model and stop words
	features_of_possible_senses = [[w for w in feature_set if w not in stop_words and w in glove_model.vocab] for feature_set in features_of_possible_senses]
	#Stemming
	#features_of_possible_senses = [[porter_stemmer.stem(w) for w in feature_set] for feature_set in features_of_possible_senses]

	results = []
	for i, instance in enumerate(instances):
		#sent = (w[0] for w in instance.context)
		#Stemming
		#sent = (porter_stemmer.stem(w[0]) for w in instance.context)
		#Remove punctuation
		sent = (re.sub(r'[^\w\s]', '', w[0]) for w in instance.context)
		#Remove stop words and words not in the model
		sent = [w for w in sent if w not in stop_words and w in glove_model.vocab]
		#sent = [w for w in sent if w in glove_model.vocab]
		results.append(_sense_for_sentence(sent, features_of_possible_senses, senses_unique))
		
		if i % 1000 == 0: print(target_word, i)

	X = np.array(results)


	accuracy_improved_lesk = (X == y).mean()
	print(accuracy_improved_lesk)

	from sklearn.metrics import confusion_matrix
	print(confusion_matrix(X, y))

	print("F1-score: {}".format(f1_score(y, X, average="macro")))

	return [accuracy_improved_lesk, precision_score(y, X, average="macro"), recall_score(y, X, average="macro"),
			f1_score(y, X, average="macro")]


final_results = []
for word in ["hard", "interest", "line", "serve"]:
	final_results.append(word_embeddings(word))

final_results = np.array(final_results)
print("Final precision: {}".format(final_results[:, 3].mean()))