from nltk.classify import NaiveBayesClassifier

from forum_profile import ForumProfile
from forum_corpus_reader import ForumCorpusReader

DATA_DIR = 'data/'
N_GRAMS = 1


def generate_features(file, label):
    # corpus = ForumCorpusReader(DATA_DIR, file)
    # profile = ForumProfile(corpus, N_GRAMS)
    # return profile.features()
    return [({file: x}, label) for x in range(10)]


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
    test_set = data_set[int(0.7 * data_size):]  # TODO: Remove label

    classifier = NaiveBayesClassifier.train(train_set)
    print(classifier.classify(test_set[:]))


if __name__ == '__main__':
    train_and_classify()
