from nltk.util import ngrams
from nltk.corpus import stopwords

# Prototyped in notebook 0.2
class ForumProfile():
    def __init__(self, forum_corpus, ngram_order):
        self.corpus = forum_corpus
        self.ngram_order = ngram_order
        self.ngrams = self.make_ngrams()
        self.ngram_distribution = FreqDist(self.ngrams)

    def make_ngrams(self):
        # Reject words that are not alphabetical and reject stopwords
        swedish_stopwords = stopwords.words('swedish')
        corpus_words = [word.lower() for word in self.corpus.words()
                        if (word.isalpha()
                        and word not in swedish_stopwords)]
        return list(ngrams(corpus_words, self.ngram_order))

if __name__ == '__main__':
    # Testing
    from forum_corpus_reader import ForumCorpusReader
    fam_corpus = ForumCorpusReader('data/', r'familjeliv-sexsamlevnad.xml')
    fam_profile = ForumProfile(fam_corpus, 3)
    print("Finished")
