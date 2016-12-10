from nltk.corpus.reader.xmldocs import XMLCorpusView
from nltk.corpus.reader.xmldocs import XMLCorpusReader
import re

# Prototyped in notebook 0.1
class ForumCorpusReader(XMLCorpusReader):
    def __init__(self, root, fileid):
        self.path = root + fileid
        XMLCorpusReader.__init__(self, root, fileid)

    def sentences(self):
        """Returns a list of sentences where each sentence is a list of words
        """
        sents = XMLCorpusView(self.path, '.*/sentence')
        sent_list = list()
        for sentence in sents:
            word_list = [word.text for word in sentence]
            sent_list.append(word_list)
        return sent_list

    def get_lemma(self, word_data):
        '''Return lemma as a string, if the lemma is already in 'grundform' then return the word text
        if the word has multiple lemmas, then return the first
        '''

        lemma = word_data.attrib['lemma']
        if lemma == "|":
            return word_data.text
        else:
            return re.search('\|*(\w*)\|+', lemma).group(0).replace("|", "")

    def tagged_words(self, lemmatize=True):
        words = XMLCorpusView(self.path, '.*/w')
        if lemmatize:
            word_tags = [ ( word.text,
                            word.attrib['pos'],
                            self.get_lemma(word) )
                         for word in words ]
        else:
            word_tags = [(word.text, word.attrib['pos']) for word in words]
        return word_tags

    def tagged_sentences(self, lemmatize=True):
        sents = XMLCorpusView(self.path, '.*/sentence')
        sent_list = list()
        for sent in sents:
            if lemmatize:
                word_list = [ ( word.text,
                                word.attrib['pos'],
                                self.get_lemma(word) )
                             for word in sent ]
            else:
                word_list = [ ( word.text,
                                word.attrib['pos'] )
                                for word in sent ]
            sent_list.append(word_list)
        return sent_list


if __name__ == '__main__':
    fam_corpus = ForumCorpusReader('data/', 'familjeliv-sex25.xml')
    sents = fam_corpus.tagged_sentences()[:25]
    print(sents)
