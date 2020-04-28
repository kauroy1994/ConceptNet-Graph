from multiprocessing import cpu_count
from joblib import Parallel,delayed
from tqdm import tqdm
from Util import IO,NLP,Conceptnet

def q(word):
    """INPUT: Takes word
       writes embedding to file
    """
    
    #word embeddings txt file
    embedding_file = 'word_embeddings.txt'

    #get word embedding from conceptnet
    embedding = Conceptnet.embed_word(word)

    #write embedding and word to file
    IO.write_to_file(embedding_file,word+':'+str(embedding))

def main():
    """main method
    """

    #use all cpu cores for parallelizing
    nc = cpu_count()

    #read 501 posts contained in csv file
    posts = IO.read_csv('data/posts.csv',501)

    #extract just the text strings
    ps = [(','.join(p.split(',')[1:-1]))[3:-3] for p in posts]

    #remove stop words
    ps_wo_stopwords = [NLP.remove_stop_words(s) for s in ps]

    #get unique words
    u_words = tqdm(NLP.get_unique_words(ps_wo_stopwords))

    #calculate number of unique words
    n_u_words = len(u_words)

    #embed unique words using conceptnet parallelized
    embeddings = Parallel(n_jobs = nc)(delayed(q)(u) for u in u_words)

if __name__ == '__main__':
    main()
