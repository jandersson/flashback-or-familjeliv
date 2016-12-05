from nltk.util import ngrams

# Prototyped in notebook 0.2
class ForumProfile():
    def __init__(self, forum_corpus, ngram_order):
        self.corpus = forum_corpus
        self.ngram_order = ngram_order
        self.ngrams = self.make_ngrams()
        self.ngram_distribution = {}
        self.count_ngrams()

    def make_ngrams(self):
        corpus_words = self.corpus.words()
        return list(ngrams(corpus_words, self.ngram_order))

    def count_ngrams(self):
        if self.ngrams is None:
            return

        for ngram in self.ngrams:
          if not ngram in self.ngram_distribution:
              self.ngram_distribution.update({ngram:1})
          else:
              ngram_occurrences = self.ngram_distribution[ngram]
              self.ngram_distribution.update({ngram:ngram_occurrences+1})
