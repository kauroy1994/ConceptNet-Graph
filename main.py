from multiprocessing import cpu_count
from joblib import Parallel,delayed
from Util import IO,NLP,Conceptnet

def main():
    """main method
    """

    #use all cpu cores for parallelizing
    nc = cpu_count()

    #create shorthand for Conceptnet.embedword
    q = Conceptnet.embed_word

    #read one post
    posts = IO.read_csv('data/posts.csv',1)

    #extract just the text strings
    ps = [(','.join(p.split(',')[1:-1]))[3:-3] for p in posts]

    #remove stop words
    ps_wo_stopwords = [NLP.remove_stop_words(s) for s in ps]

    #get unique words
    u_words = (NLP.get_unique_words(ps_wo_stopwords))

    #embed unique words using conceptnet parallelized
    embeddings = Parallel(n_jobs = nc)(delayed(q)(u) for u in u_words)
    

if __name__ == '__main__':
    main()
