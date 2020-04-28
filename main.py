from Util import IO,NLP,Conceptnet

def main():
    """main method
    """

    #read one post
    posts = IO.read_csv('data/posts.csv',1)

    #extract just the text strings
    ps = [(','.join(p.split(',')[1:-1]))[3:-3] for p in posts]

    #remove stop words
    ps_wo_stopwords = [NLP.remove_stop_words(s) for s in ps]

    #get unique words
    u_words = NLP.get_unique_words(ps_wo_stopwords)

    #embed unique words using conceptnet
    print (Conceptnet.embed_word(u_words[10]))
    

if __name__ == '__main__':
    main()
