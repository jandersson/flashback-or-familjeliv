from nltk import classify
from nltk.classify import NaiveBayesClassifier
import random
import pickle
import os.path

from forum_profile import ForumProfile
from forum_corpus_reader import ForumCorpusReader

DATA_DIR = 'data/'
FAM_FILE = r'familjeliv-sex1.xml'
FLASH_FILE = r'flashback-sex1.xml'
NGRAM_ORDER = 1
TRAIN_TEST_RATIO = 0.7


def generate_features(file, label):
    corpus = ForumCorpusReader(DATA_DIR, file)
    profile = ForumProfile(corpus, label, NGRAM_ORDER)
    return profile.generate_features()


def load_or_generate_features(file, label, load_if_exists=True):
    pickle_file = "data/{label}.p".format(label=label)
    if os.path.isfile(pickle_file) and load_if_exists:
        return pickle.load(open(pickle_file, "rb"))
    feats = generate_features(file, label)
    pickle.dump(feats, open(pickle_file, "wb"))
    return feats


def train_and_classify():
    fam_set = load_or_generate_features(FAM_FILE, 'fam')
    flash_set = load_or_generate_features(FLASH_FILE, 'flash')
    data_set = fam_set + flash_set
    random.shuffle(data_set)

    data_size = len(data_set)
    train_set = data_set[:int(TRAIN_TEST_RATIO * data_size)]
    test_set = data_set[int(TRAIN_TEST_RATIO * data_size):]
    test_set_unlabeled = [features for (features, labels) in test_set]

    classifier = NaiveBayesClassifier.train(train_set)
    train_accuracy = classify.accuracy(classifier, train_set)
    test_accuracy = classify.accuracy(classifier, test_set)

    print("Train accuracy: {acc}".format(acc=train_accuracy))
    print("Test accuracy: {acc}".format(acc=test_accuracy))

    print("----------------")

    # Print probabilities
    for i, test in enumerate(test_set_unlabeled[:10]):
        guess = classifier.classify(test)
        fam_prob = classifier.prob_classify(test).prob('fam')
        flash_prob = classifier.prob_classify(test).prob('flash')

        print("Data point: {data} \nGuess: {guess}".format(data=test_set[i], guess=guess))
        print("Fam probability: {fam} \nFlash probability: {flash}".format(fam=fam_prob, flash=flash_prob))
        print("\n")


if __name__ == '__main__':
    train_and_classify()
