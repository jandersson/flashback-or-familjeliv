from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist


# Prototyped in notebook 0.2
class ForumProfile():
    def __init__(self, forum_corpus, classname, ngram_order=1):
        self.corpus = forum_corpus
        self.classname = classname
        self.ngram_order = ngram_order
        # self.words = self.preprocess()
        # self.ngrams = self.make_ngrams()
        # self.ngram_distribution = FreqDist(self.ngrams)

    def generate_features(self, tagged=False):
        '''If the dataset is tagged then we will use the lemma instead of the word, which is the third element of the data tuple
        '''
        feature_list = list()
        if not tagged:
            for sentence in self.corpus.sentences():
                features = FreqDist(sentence)
                data_vec = (features, self.classname)
                feature_list.append(data_vec)
        else:
            for sentence in self.corpus.tagged_sentences():
                features = FreqDist([tup[2] for tup in sentence])
                data_vec = (features, self.classname)
                feature_list.append(data_vec)

        return feature_list

    def preprocess(self):
        # Reject words that are not alphabetical and reject stopwords
        swedish_stopwords = stopwords.words('swedish')
        corpus_words = [word.lower() for word in self.corpus.words()
                        if (word.isalpha()
                        and word not in swedish_stopwords)]
        return FreqDist(corpus_words)

if __name__ == '__main__':
    # Testing
    from forum_corpus_reader import ForumCorpusReader
    fam_corpus = ForumCorpusReader('data/', r'familjeliv-sex25.xml')
    fam_profile = ForumProfile(fam_corpus, 1)
    print(fam_profile.generate_features()[:5])
    # fam_profile.corpus.tagged_words()

    print("Finished")
