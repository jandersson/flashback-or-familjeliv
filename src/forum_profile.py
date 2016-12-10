from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier

# Prototyped in notebook 0.2
class ForumProfile():
    def __init__(self, forum_corpus, classname, ngram_order=1):
        self.corpus = forum_corpus
        self.classname = classname
        self.ngram_order = ngram_order
        self.stopwords  = stopwords.words('swedish')
        # self.words = self.preprocess()
        # self.ngrams = self.make_ngrams()
        # self.ngram_distribution = FreqDist(self.ngrams)

    def generate_features(self, use_lemmas=False, do_preprocess=True):
        '''If the dataset is tagged then we will use the lemma instead of the word, which is the third element of the data tuple
        '''
        feature_list = list()
        if not use_lemmas:
            for sentence in self.corpus.sentences():
                if do_preprocess:
                    sentence = self.preprocess(sentence)
                features = FreqDist(sentence)
                data_vec = (features, self.classname)
                feature_list.append(data_vec)
        else:
            for sentence in self.corpus.tagged_sentences():
                if do_preprocess:
                    sentence = self.preprocess(sentence)
                features = FreqDist([tup[2] for tup in sentence])
                data_vec = (features, self.classname)
                feature_list.append(data_vec)

        return feature_list

    def preprocess(self, sentence):
        '''Takes a list of words/tokens and returns a new list with the stop words and non-alphabetical tokens removed. Lowercases all words.
        '''
        proc_sentence = [word.lower() for word in sentence
                        if (word.isalpha()
                        and word not in self.stopwords)]
        return proc_sentence

if __name__ == '__main__':
    # Testing
    import os

    #
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    from forum_corpus_reader import ForumCorpusReader
    fam_corpus = ForumCorpusReader('../data/', r'familjeliv-sex25.xml')
    fam_profile = ForumProfile(fam_corpus, classname = 1)
    print(fam_profile.generate_features(do_preprocess = True))
    # fam_profile.corpus.tagged_words()

    print("Finished")
