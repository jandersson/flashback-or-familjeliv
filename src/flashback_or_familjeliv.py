from nltk import classify
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier
import random
import pickle
import os.path
import csv
from datetime import datetime

from forum_profile import ForumProfile
from forum_corpus_reader import ForumCorpusReader

DATA_DIR = 'data/'
FAM_DIR = 'familjeliv/'
FLASH_DIR = 'flashback/'

FLASH_LABEL = 'flash'
FL_LABEL = 'fam'

NGRAM_ORDER = 1
TRAIN_TEST_RATIO = 0.7
MAXENT_CUTOFF = 2

TIMESTAMP = str(datetime.now())
COMMENT = "Familjeliv Housing vs Familjeliv Sexsamlivet"

def write_results(results):
    '''Writes results to csv file. Takes a list of results.
    '''
    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)

def write_informative_feats(classifier, num_feats):
    '''Takes a classifier and writes out the given number of informative features to disk
    '''
    if classifier.__class__.__name__ != "NaiveBayesClassifier":
        return

    with open('feats.txt', 'a') as f:
        f.write(str(classifier.most_informative_features(num_feats)) + " " + COMMENT + '\n')

def generate_features(class_dir, file, label):
    """
    Generate and return labeled features of DATA_DIR/class_dir/file with
    """
    corpus = ForumCorpusReader(DATA_DIR + class_dir, file)
    profile = ForumProfile(corpus, label, NGRAM_ORDER)
    return profile.generate_features()


def load_or_generate_features(class_dir, label, load_if_exists=True):
    """
    For each .xml file in DATA_DIR/class_dir, load the corresponding pickle file if exists, otherwise create it.
    """
    # pickle_file = "data/{label}.p".format(label=label)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    feature_list = []
    for filename in os.listdir(DATA_DIR + class_dir):
        if filename.endswith(".xml"):
            pickle_file = DATA_DIR + class_dir + filename[:-4] + ".p"
            if not (os.path.isfile(pickle_file) and load_if_exists):
                print("Generating features from: {filename}".format(filename=filename))
                feats = generate_features(class_dir, filename, label)
                pickle.dump(feats, open(pickle_file, "wb"))
            print("Loading features from: {pickle_file}".format(pickle_file=pickle_file))
            feature_list += pickle.load(open(pickle_file, "rb"))
    return feature_list


def train_and_classify(classifierClass):
    # Load data
    fam_set = load_or_generate_features(FAM_DIR, FL_LABEL)
    flash_set = load_or_generate_features(FLASH_DIR, FLASH_LABEL)
    data_set = fam_set + flash_set
    random.shuffle(data_set)

    # Split into training and test set
    data_size = len(data_set)
    train_set = data_set[:int(TRAIN_TEST_RATIO * data_size)]
    test_set = data_set[int(TRAIN_TEST_RATIO * data_size):]
    test_set_unlabeled = [features for (features, labels) in test_set]

    # Create classifier

    print("Training {classifier}...".format(classifier=classifierClass.__name__))
    if classifierClass.__name__ == "MaxentClassifier":
        classifier = classifierClass.train(train_set, max_iter=MAXENT_CUTOFF)
    else:
        classifier = classifierClass.train(train_set)
    train_accuracy = classify.accuracy(classifier, train_set)
    test_accuracy = classify.accuracy(classifier, test_set)

    # Print classifier statistics
    print("Train accuracy: {acc}".format(acc=train_accuracy))
    print("Test accuracy: {acc}".format(acc=test_accuracy))
    print("----------------")

    print(classifier.show_most_informative_features(10))
    # Print classification with probabilities for 10 test data
    for i, test in enumerate(test_set_unlabeled[:10]):
        guess = classifier.classify(test)
        fam_prob = classifier.prob_classify(test).prob(FL_LABEL)
        flash_prob = classifier.prob_classify(test).prob(FLASH_LABEL)

        print("Data point: {data} \nGuess: {guess}".format(data=test_set[i], guess=guess))
        print("Fam probability: {fam} \nFlash probability: {flash}".format(fam=fam_prob, flash=flash_prob))
        print("\n")
    write_results([TIMESTAMP, classifierClass.__name__, train_accuracy, test_accuracy, COMMENT])
    write_informative_feats(classifier, 10)

if __name__ == '__main__':
    # train_and_classify(NaiveBayesClassifier)
    # train_and_classify(DecisionTreeClassifier)
    train_and_classify(MaxentClassifier)
