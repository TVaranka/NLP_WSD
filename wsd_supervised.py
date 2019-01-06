# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import random
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from collections import defaultdict

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from nltk.wsd import lesk
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import Counter

# The following shows how the senseval corpus consists of instances, where each instance
# consists of a target word (and its tag), it position in the sentence it appeared in
# within the corpus (that position being word position, minus punctuation), and the context,
# which is the words in the sentence plus their tags.
#
# senseval.instances()[:1]
# [SensevalInstance(word='hard-a', position=20, context=[('``', '``'), ('he', 'PRP'),
# ('may', 'MD'), ('lose', 'VB'), ('all', 'DT'), ('popular', 'JJ'), ('support', 'NN'),
# (',', ','), ('but', 'CC'), ('someone', 'NN'), ('has', 'VBZ'), ('to', 'TO'),
# ('kill', 'VB'), ('him', 'PRP'), ('to', 'TO'), ('defeat', 'VB'), ('him', 'PRP'),
# ('and', 'CC'), ('that', 'DT'), ("'s", 'VBZ'), ('hard', 'JJ'), ('to', 'TO'), ('do', 'VB'),
# ('.', '.'), ("''", "''")], senses=('HARD1',))]

def senses(word):
    """
    This takes a target word from senseval-2 (find out what the possible
    are by running senseval.fileides()), and it returns the list of possible 
    senses for the word
    """
    return list(set(i.senses[0] for i in senseval.instances(word)))

# Both above and below, we depend on the (non-obvious?) fact that although the field is
#  called 'senses', there is always only 1, i.e. there is no residual ambiguity in the
#  data as we have it

def sense_instances(instances, sense):
    """
    This returns the list of instances in instances that have the sense
    """
    return [instance for instance in instances if instance.senses[0]==sense]

# >>> sense3 = sense_instances(senseval.instances('hard.pos'), 'HARD3')
# >>> sense3[:2]
# [SensevalInstance(word='hard-a', position=15,
#  context=[('my', 'PRP$'), ('companion', 'NN'), ('enjoyed', 'VBD'), ('a', 'DT'), ('healthy', 'JJ'), ('slice', 'NN'), ('of', 'IN'), ('the', 'DT'), ('chocolate', 'NN'), ('mousse', 'NN'), ('cake', 'NN'), (',', ','), ('made', 'VBN'), ('with', 'IN'), ('a', 'DT'), ('hard', 'JJ'), ('chocolate', 'NN'), ('crust', 'NN'), (',', ','), ('topping', 'VBG'), ('a', 'DT'), ('sponge', 'NN'), ('cake', 'NN'), ('with', 'IN'), ('either', 'DT'), ('strawberry', 'NN'), ('or', 'CC'), ('raspberry', 'JJ'), ('on', 'IN'), ('the', 'DT'), ('bottom', 'NN'), ('.', '.')],
#  senses=('HARD3',)),
#  SensevalInstance(word='hard-a', position=5,
#  context=[('``', '``'), ('i', 'PRP'), ('feel', 'VBP'), ('that', 'IN'), ('the', 'DT'), ('hard', 'JJ'), ('court', 'NN'), ('is', 'VBZ'), ('my', 'PRP$'), ('best', 'JJS'), ('surface', 'NN'), ('overall', 'JJ'), (',', ','), ('"', '"'), ('courier', 'NNP'), ('said', 'VBD'), ('.', '.')],
# senses=('HARD3',))]


_inst_cache = {}

STOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 'also', 'an', 'any',
             'are', 'as', 'at', 'and', 'be', 'being', 'because', 'been', 'but', 'by',
             'can', "'d", 'did', 'do', "don'", 'don', 'for', 'from', 'had','has', 'have', 'he',
             'her','him', 'his', 'how', 'if', 'is', 'in', 'it', 'its', "'ll", "'m", 'me',
             'more', 'my', 'n', 'no', 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s",
             'said', 'say', 'says', 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the',
             'them', 'they', 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you',
             'your']

STOPWORDS_SET=set(STOPWORDS)

NO_STOPWORDS = []

def wsd_context_features(instance, vocab, dist=3):
    features = {}
    ind = instance.position
    con = instance.context
    for i in range(max(0, ind-dist), ind):
        j = ind-i
        features['left-context-word-%s(%s)' % (j, con[i][0])] = True

    for i in range(ind+1, min(ind+dist+1, len(con))):
        j = i-ind
        features['right-context-word-%s(%s)' % (j, con[i][0])] = True

 
    features['word'] = instance.word
    features['pos'] = con[1][1]
    return features

def wsd_word_features(instance, vocab, dist=3):
    """
    Create a featureset where every key returns False unless it occurs in the
    instance's context
    """
    features = defaultdict(lambda:False)
    features['alwayson'] = True
    #cur_words = [w for (w, pos) in i.context]
    try:
        for(w, pos) in instance.context:
            if w in vocab:
                features[w] = True
    except ValueError:
        pass
    return features

def extract_vocab_frequency(instances, stopwords=STOPWORDS_SET, n=300):
    """
    Given a list of senseval instances, return a list of the n most frequent words that
    appears in its context (i.e., the sentence with the target word in), output is in order
    of frequency and includes also the number of instances in which that key appears in the
    context of instances.
    """
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        words = (c[0] for c in i.context if not c[0] == target)
        for word in set(words) - set(stopwords):
            fd[word] += 1
            #for sense in i.senses:
                #cfd[sense][word] += 1
    return fd.most_common()[:n+1]
        
def extract_vocab(instances, stopwords=STOPWORDS_SET, n=300):
    return [w for w,f in extract_vocab_frequency(instances,stopwords,n)]
    
def plotConfusionMatrix(confusionmatrix, classes,model_name):
    """
    This function plots confusionmatrix in heat-map format
    """
    norm_conf = []
    for i in confusionmatrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, interpolation='nearest')
    width, height = confusionmatrix.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(confusionmatrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
    cb = fig.colorbar(res)
    plt.xticks(range(width), classes[:width])
    plt.yticks(range(height), classes[:height])
    plt.title("Confusion matrix for {}".format(model_name))
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")

    
def wst_classifier(trainer, word, features, stopwords_list = NO_STOPWORDS, number=300, distance=3, confusionmatrix=False, train_size=0.8, metric='f1score', model_name=None):
    """
    This function takes as arguments:
        a trainer (e.g., NaiveBayesClassifier.train);
        a target word from senseval2 (you can find these out with senseval.fileids(),
            and they are 'hard.pos', 'interest.pos', 'line.pos' and 'serve.pos');
        a feature set (this can be wsd_context_features or wsd_word_features);
        a number (defaults to 300), which determines for wsd_word_features the number of
            most frequent words within the context of a given sense that you use to classify examples;
        a distance (defaults to 3) which determines the size of the window for wsd_context_features (if distance=3, then
            wsd_context_features gives 3 words and tags to the left and 3 words and tags to
            the right of the target word);
        train_size, which determines as decimal number how many samples is chosen for training from original dataset;
        metric, which determines the used metric for evaluation, 'f1score' is default and returns only f1-score and 'all' returns accuracy,
        precision, recall and f1score in a list;
        parameter model_name is needed when printing confusionmatrix;
        parameter returnmodel is needed when function should return only the trained classifier;

    Calling this function splits the senseval data for the word into a training set and a test set (the way it does
    this is the same for each call of this function, because the argument to random.seed is specified,
    but removing this argument would make the training and testing sets different each time you build a classifier).

    It then trains the trainer on the training set to create a classifier that performs WSD on the word,
    using features (with number or distance where relevant).

    It then tests the classifier on the test set, and prints its accuracy on that set.

    If confusion_matrix==True, then calling this function prints out a confusion matrix, where each cell [i,j]
    indicates how often label j was predicted when the correct label was i (so the diagonal entries indicate labels
    that were correctly predicted).
    """
    #print("Reading data...")
    global _inst_cache
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    #print(' Senses: ' + ' '.join(senses))

    # Split the instances into a training and test set,
    #if n > len(events): n = len(events)
    n = len(events)
    random.seed(5444522)
    random.shuffle(events)
    training_data = events[:int(train_size * n)] #maximum train_size is 0.8
    test_data = events[int(0.8 * n):n]
    # Train classifier
    classifier = trainer([(features(i, vocab, distance), label) for (i, label) in training_data])
    gold = [label for (i, label) in test_data]
    derived = [classifier.classify(features(i,vocab,distance)) for (i,label) in test_data]
    classes = list(set(gold))
    for j in range(len(gold)):
        gold[j] = classes.index(gold[j])
        derived[j] = classes.index(derived[j])
    # Test classifier
    f1score = metrics.f1_score(gold, derived, average='macro')
    if confusionmatrix==True:
        cm = confusion_matrix(gold,derived)
        plotConfusionMatrix(cm,classes, model_name)
        return
    if metric=='all':
        prec = metrics.precision_score(gold, derived, average='macro')
        rec = metrics.recall_score(gold, derived, average='macro')
        acc = metrics.accuracy_score(gold, derived)
        output = np.asarray([acc, prec, rec, f1score])
        return output, gold, derived
    return f1score

def evaluation_for_all_targetwords(targetwords, classifier, features, trainsize=0.8, windowsize=2, mostfrequentwords=400, mode='number', usedmetric='f1score'):
    """
    This function calculates averaged F1-score for based on used targetwords, classifier and features.
    There are 3 different modes: 'number', 'list' or 'extended_list', which affects to returned object.
    """
    if mode=='number':
        temp_acc = 0
        for i in range(len(targetwords)):
            temp_acc += wst_classifier(classifier, targetwords[i]+'.pos', features, number=mostfrequentwords ,distance=windowsize, train_size=trainsize, metric=usedmetric)
        return temp_acc/len(targetwords)
    if mode=='list':
        accuracies = []
        for	i in range(len(targetwords)):
            accuracies.append(wst_classifier(classifier, targetwords[i]+'.pos', features, number=mostfrequentwords, distance=windowsize, train_size=trainsize, metric=usedmetric))  
        return accuracies
    if mode=='extended_list':
        temp_acc = np.zeros(4)
        gold_list = []
        derived_list = []
        for i in range(len(targetwords)):
            temp_metrics, gold, derived = wst_classifier(classifier, targetwords[i] + '.pos', features, metric=usedmetric, distance=5)
            temp_acc = np.sum([temp_metrics, temp_acc], axis=0)
            gold_list.append(gold)
            derived_list.append(derived)
        return temp_acc/len(targetwords), gold_list, derived_list
		
def matplotlib_bar(targetwords, y_values, loc1, ylabelname, titlename, usedcolor='b'):
    """
    This function plots bar chart for given parameters
    """
    plt.figure()
    plt.bar(targetwords, y_values, align='center', alpha=0.5, color=usedcolor)
    plt.xticks(targetwords)
    plt.ylabel(ylabelname)
    plt.title(titlename)
    xpos = np.arange(len(y_values))
    for x, y in zip(xpos, y_values):
        plt.text(x, y + loc1, '%.2f' % y, ha='center', va= 'bottom')

def matplotlib_graph(x_values, y_values, xlabelname, ylabelname, titlename):
    """
    This function plots graph for given parameters
    """	
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel(xlabelname)
    plt.ylabel(ylabelname)
    plt.title(titlename)
	
def generate_ensemble_predictions(targetwords, indices, gold):
    """
    This function performs voting algorithm from a set of classified test samples with different classifiers
    If the voting result is equal, it choose ramdomly predicted class
    """
    gold_set = []
    for i in indices:
        gold_set.append(gold[i])
    dataset = list(map(list, zip(*gold_set)))
    ensemble_predictions = []
    for i in range(len(targetwords)):
        voting_data = dataset[i]
        voting_data_t = list(map(list, zip(*voting_data)))
        ensemble_temp = []
        for j in range(len(voting_data_t)):
            b = Counter(voting_data_t[j])
            ensemble_temp.append(b.most_common(1)[0][0])
        ensemble_predictions.append(ensemble_temp)
    return ensemble_predictions
        
	  
def demo():
    #################### Task 1 ###########################
    print(senseval.fileids())
    target_words = ['hard', 'interest', 'line', 'serve']
    # Number of samples for each senses of target words
    for j in range(len(target_words)):
        hard_sense_fd = nltk.FreqDist([i.senses[0] for i in senseval.instances(target_words[j]+'.pos')])
        print(hard_sense_fd.most_common())	
	
    # First check in pie charts how samples are divided to classes for each target word
    for j in range(len(target_words)):
        plt.figure()
        hard_sense_fd = nltk.FreqDist([i.senses[0] for i in senseval.instances(target_words[j]+'.pos')])
        senses = list(hard_sense_fd.keys())
        values = []
        for k in range(len(senses)):
            values.append(hard_sense_fd.freq(senses[k]))
        plt.pie(values, labels=senses, autopct='%1.1f%%', shadow=False, startangle=180)
        plt.axis('equal')
        plt.title('Pie chart for the target word "{}"'.format(target_words[j]))

    # Train Naive bayes classifier with window based method and frequency based method
    print("NB, with features based on 300 most frequent context words")
    acc1 = evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_word_features)
    print('F1-score: %6.4f' % acc1)
    print()
    print("NB, with features based word + pos in 6 word window")
    acc1 = evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_context_features)
    print('F1-score: %6.4f' % acc1)     
    # window based method produce better result than frequency based method

    # Barplots of F1-score when classifier chooses class randomly
    classifier_dummy = SklearnClassifier(DummyClassifier()).train
    accuracies0 = evaluation_for_all_targetwords(target_words, classifier_dummy, wsd_context_features, mode='list')
    matplotlib_bar(target_words, accuracies0, 0.00001, 'F1-score', 'F1-score when samples are classified randomly to different classes')
	
    # Barplots of F1-score for all target words with naive bayes classifier
    accuracies1 = evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_context_features, mode='list')
    matplotlib_bar(target_words, accuracies1, 0.00001, 'F1-score', 'F1-scores for NB-classifier with different target words') 
    # as a result word 'hard' has best performance
	
    # Graph of F1-score with different size of training set (wsd_context_features) (the size of testset is always 0.2)
    size_of_trainset = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    accuracies2 = []
    for i in range(len(size_of_trainset)):
        accuracies2.append(evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_context_features, mode='number', trainsize=size_of_trainset[i]))
    matplotlib_graph(size_of_trainset, accuracies2, 'Size of training set', 'F1-score', 'F1-scores for NB-classifier with different trainingset sizes')
    # with training set size 0.8 we get best result
	
    # Graph of F1-score with different size of the window (wsd_context_features)
    size_of_window = [1,2,3,4,5,6,7,8,9,10,13,16,20]
    size_of_win = ["3","5","7","9","11","13","15","17","19","21","27","33","41"]	
    accuracies3 = []	
    for i in range(len(size_of_window)):
        accuracies3.append(evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_context_features, mode='number', windowsize=size_of_window[i]))
    matplotlib_graph(size_of_window, accuracies3, 'Size of window', 'F1-score', 'F1-scores for NB-classifier with different window sizes')	
    plt.xticks(size_of_window,size_of_win)
    # we get best result with window size 5

    # Last changing parameter is the number of most frequent words for wsd_word_features
    num_most_freq_words = [5,10,30,50,100,150,200,300,400,500,700]
    accuracies4 = []	
    for i in range(len(num_most_freq_words)):
        accuracies4.append(evaluation_for_all_targetwords(target_words, NaiveBayesClassifier.train, wsd_word_features, mode='number', mostfrequentwords=num_most_freq_words[i]))
    matplotlib_graph(num_most_freq_words, accuracies4, 'Number of most frequent words', 'F1-score', 'F1-scores for NB-classifier with different number of most frequent words')	
    plt.xticks(num_most_freq_words)
    locs1, labels1 = plt.xticks()
    plt.setp(labels1, rotation=45)
    # best number of words is 300-500
	
    #################### Task 2 ###########################
    # Used classifiers for task 2
    classifier_knn = SklearnClassifier(KNeighborsClassifier()).train
    classifier_linear_svm = SklearnClassifier(SVC(kernel='linear')).train
    classifier_poly_svm = SklearnClassifier(SVC(kernel='poly', gamma=0.1, C=1)).train
    classifier_rbf_svm = SklearnClassifier(SVC(kernel='rbf', gamma=0.1, C=1)).train
    classifier_rf = SklearnClassifier(RandomForestClassifier()).train
    classifier_ab = SklearnClassifier(AdaBoostClassifier()).train
    classifier_xgb = SklearnClassifier(xgb.XGBClassifier()).train
    classifier_nb = NaiveBayesClassifier.train
    classifier_me = MaxentClassifier.train

    classifiers = [classifier_poly_svm, classifier_ab, classifier_knn, classifier_xgb, classifier_nb, classifier_rf, classifier_rbf_svm, classifier_me, classifier_linear_svm]
    classifiernames = ['Polynomial SVM', 'Adaptive boosting', 'K-nearest neighbors', 'XGBoost', 'Naive-bayes', 'Random Forest','RBF SVM', 'Maximum entropy', 'Linear SVM']
    # Compare the performance of classifier with accuracy, precision, recall and f1score values
    accuracies, precisions, recalls, f1scores, values, gold_list, derived_list = [],[],[],[],[],[],[]

    for i in range(len(classifiers)):
        values1, gold, derived = evaluation_for_all_targetwords(target_words, classifiers[i], wsd_context_features, mode='extended_list', usedmetric='all')
        values.append(values1)
        gold_list.append(gold)
        derived_list.append(derived)

    for i in range(len(values)):
        accuracies.append(values[i][0])
        precisions.append(values[i][1])
        recalls.append(values[i][2])
        f1scores.append(values[i][3])

    # Barplots of accuracies for all target words and all classifiers
    matplotlib_bar(classifiernames, accuracies, 0.001, 'Accuracies', 'Bar plot of accuracies for different classifiers', usedcolor='b')
    locs1, labels1 = plt.xticks()
    plt.setp(labels1, rotation=45)
	
    # Barplots of precisions for all target words and all classifiers
    matplotlib_bar(classifiernames, precisions, 0.001, 'Precisions', 'Bar plot of precision values for different classifiers', usedcolor='r')
    locs1, labels1 = plt.xticks()
    plt.setp(labels1, rotation=45)

    # Barplots of recalls for all target words and all classifiers
    matplotlib_bar(classifiernames, recalls, 0.001, 'Recalls', 'Bar plot of recall values for different classifiers', usedcolor='g')
    locs1, labels1 = plt.xticks()
    plt.setp(labels1, rotation=45)

    # Barplots of F1-scores for all target words and all classifiers
    matplotlib_bar(classifiernames, f1scores, 0.001, 'F1-scores', 'Bar plot of F1-score values for different classifiers', usedcolor='k')
    locs1, labels1 = plt.xticks()
    plt.setp(labels1, rotation=45)

    # Ensemble classifier by choosing 3 and 5 best classifiers from all classifiers which choose the majority vote when predicting
    # Five best classifiers are based on F1-score: Linear SVM, Maximum entropy, RBF SVM, Naive Bayes classifier and XGBoost (indexes: 3,4,6,7,8)
    # Three best classifiers are: Linear SVM, Maximum entropy and RBF SVM (indexes: 6,7,8)
    five_best_predictions = generate_ensemble_predictions(target_words, [3,4,6,7,8], derived_list)
    three_best_predictions = generate_ensemble_predictions(target_words, [6,7,8], derived_list)

    temp_f1 = 0
    temp_acc = 0
    for i in range(len(target_words)):
        temp_f1 += metrics.f1_score(five_best_predictions[i], gold_list[0][i], average='macro')
        temp_acc += metrics.accuracy_score(five_best_predictions[i], gold_list[0][i])
    val1 = temp_f1/len(target_words)
    val2 = temp_acc/len(target_words)
    print('Accuracy for five-best ensemble classifier: %6.4f\n' % val2)
    print('F1-score for five-best ensemble classifier: %6.4f\n' % val1)
	
    temp_f1 = 0
    temp_acc = 0
    for i in range(len(target_words)):
        temp_f1 += metrics.f1_score(three_best_predictions[i], gold_list[0][i], average='macro')
        temp_acc += metrics.accuracy_score(three_best_predictions[i], gold_list[0][i])
    val1 = temp_f1/len(target_words)
    val2 = temp_acc/len(target_words)
    print('Accuracy for three-best ensemble classifier: %6.4f\n' % val2)
    print('F1-score for three-best ensemble classifier: %6.4f\n' % val1)
            
        
 
    """
    # Confusion matrices for all classifiers
    wst_classifier(classifier_nb, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[0])
    wst_classifier(classifier_me, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[1])
    wst_classifier(classifier_poly_svm, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[2])
    wst_classifier(classifier_linear_svm, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[3])
    wst_classifier(classifier_rbf_svm, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[4])
    wst_classifier(classifier_rf, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[5])
    wst_classifier(classifier_knn, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[6])
    wst_classifier(classifier_ab, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[7])
    wst_classifier(classifier_xgb, 'hard.pos', wsd_context_features, confusionmatrix=True, model_name=classifiers[8])
    """   
    plt.show()

demo()
