from nltk.corpus.reader.xmldocs import XMLCorpusView
from nltk.corpus.reader.xmldocs import XMLCorpusReader

# Prototyped in notebook 0.1
class ForumCorpusReader(XMLCorpusReader):
    def __init__(self, root, fileid):
        self.path = root+fileid
        XMLCorpusReader.__init__(self, root, fileid)

    def sentences(self):
        '''Returns a list of sentences where each sentence is a list of words
        '''
        sents = XMLCorpusView(self.path, '.*/sentence')
        sent_list = list()
        for sentence in sents:
            word_list = [word.text for word in sentence]
            sent_list.append(word_list)
        return sent_list

    def tagged_words(self, lemmatize=True):
        words = XMLCorpusView(self.path, '.*/w')
        if lemmatize:
            word_tags = [ (word.text,
                           word.attrib['pos'],
                           word.attrib['lemma'].replace("|", ""))
                         for word in words ]
        else:
            word_tags = [ (word.text, word.attrib['pos']) for word in words ]
        return word_tags

    def tagged_sentences(self, lemmatize=True):
        sents = XMLCorpusView(self.path, '.*/sentence')
        sent_list = list()
        for sent in sents:
            if lemmatize:
                word_list = [ (word.text,
                               word.attrib['pos'],
                               word.attrib['lemma'].replace("|", ""))
                               for word in sent ]
            else:
                word_list = [ (word.text,
                               word.attrib['pos'])
                               for word in sent ]
            sent_list.append(word_list)
        return sent_list

if __name__ == '__main__':
    fam_corpus = ForumCorpusReader('data/', 'familjeliv-sex25.xml')
    print(fam_corpus.tagged_sentences()[:2])
