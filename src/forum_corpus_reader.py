from nltk.corpus.reader.xmldocs import XMLCorpusView
from nltk.corpus.reader.xmldocs import XMLCorpusReader

# Prototyped in notebook 0.1
class ForumCorpusReader(XMLCorpusReader):
    def __init__(self, root, fileid):
        XMLCorpusReader.__init__(self, root, fileid)

    def sentences(self, fileid):
        '''This does nothing at the moment, but it would be nice to be able to return the sentences.
        This is probably possible with the XMLCorpusView, where you can specify tags of interest.
        '''
        return XMLCorpusView(fileid, 'sentences')
