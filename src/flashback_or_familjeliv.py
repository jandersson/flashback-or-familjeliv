from nltk import classify
from nltk.classify import NaiveBayesClassifier

from forum_profile import ForumProfile
from forum_corpus_reader import ForumCorpusReader

DATA_DIR = 'data/'
N_GRAMS = 1


def generate_features(file, label):
    # corpus = ForumCorpusReader(DATA_DIR, file)
    # profile = ForumProfile(corpus, N_GRAMS)
    # return profile.features()
    return [({file: 1}, label) for x in range(10)]


def train_and_classify():
    fam_file = r'familjeliv-sex1.xml'
    flash_file = r'flashback-sex1.xml'

    fam_set = generate_features(fam_file, 'fam')
    flash_set = generate_features(flash_file, 'flash')

    data_set = fam_set + flash_set
    import random
    random.shuffle(data_set)
    data_size = len(data_set)
    train_set = data_set[:int(0.7 * data_size)]
    test_set = data_set[int(0.7 * data_size):]
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
