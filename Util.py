from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from text_to_uri import standardized_uri

class NLP(object):
    """performs NLP related tasks
    """

    def __init__(self):
        """currently constructor not
           in use
        """
        pass

    @staticmethod
    def remove_stop_words(string):
        """
           INPUT: Takes a string
           removes stop words using
           nltk corpus filtering
        """

        #get nltk stop words
        stop_words = set(stopwords.words('english'))

        #tokenize string
        word_tokens = word_tokenize(string)

        #remove stop words
        return (' '.join([w for w in word_tokens if not w in stop_words]))

    @staticmethod
    def get_unique_words(strings):
        """
        INPUT: Take a set of strings
        extracts unique words
        """

        #place holder for unique words
        u = []

        #extract unique words, each string
        for string in strings:
            t = word_tokenize(string)
            string_u_words = [w for w in t if ((w not in u) and len(w)>1)]
            u += string_u_words

        return (u)

class IO(object):
    """for file I/O related ops
    """

    def __init__(self):
        """constructor not in use
        """

        pass

    @staticmethod
    def read_csv(file_to_read,n):
        """INPUT: file to read
           INPUT: num lines to read
           returns n lines as list
        """

        #place holder for n file lines
        lines = []
        
        with open(file_to_read,'r') as fp:

            #counter to keep track of lines
            c = 0

            #infinite loop to read n lines
            while True:

                #if n lines read break
                if c == n+1:
                    break
                
                line = fp.readline()
                lines.append(line)
                c += 1

                #if End of file break
                if not line:
                    break

        #remove header line
        return (lines[1:])

class Conceptnet(object):
    """wrapper for conceptnet KG
       related information extraction
    """

    def __init__(self):
        """constructor currently
           not in use
        """
        
        pass

    @staticmethod
    def search_conceptnet(word_uri):
        """INPUT: standardized word
           searches concept net for
           embedding
        """

        #search concept net embeddings
        with open('numberbatch-en.txt','r') as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                if line.split(' ')[0] == word_uri:
                    embedding = line.split(' ')[1:]
                    embedding = list(embedding)
                    return ([float(x) for x in embedding])
                
        #if word not found return false
        return False

    @staticmethod
    def embed_word(word):
        """INPUT: word to embed
           embeds word from concept net
           embeddings
        """

        #initialize embedding to 0
        word_embedding = [0.0 for i in range(300)]

        #standardize word for conceptnet lookup
        word_uri = standardized_uri('en',word).split('/')[-1]

        #try and find embedding till smallest token reached
        while len(word) > 0:
            embedding = Conceptnet.search_conceptnet(word_uri)

            #if embedding found return
            if embedding:
                return (embedding)

            #if not found reduce token and try again
            if not embedding:
                word = word[:-1]
                word_uri = standardized_uri('en',word).split('/')[-1]

        return (word_embedding)
