import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    """ Define START/ STOP marker"""
    if n>1:
        number_of_start = n-1
    else:
        number_of_start = 1

    start_marker = ('START',) * number_of_start
    stop_marker = ('STOP',)

    """Create a tuple with START/STOP MARKER and sequence strings"""
    padded_sequence = start_marker + tuple(sequence) + stop_marker

    """Return padded ngrams over list of string. The result should be a list of Python tuples"""
    ngrams = []

    for i in range(len(padded_sequence)-n+1):
        ngram = padded_sequence[i:i+n]
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        self.number_of_sentence = 0

        for sentence in corpus:
            unigrams = get_ngrams(sentence,1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence,3)
            self.number_of_sentence += 1

            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

        self.total_word_counts = 0

        for unigram in unigrams:
            if unigram != ('START' ,):
                self.total_word_counts += self.unigramcounts[unigram]


        self.size_of_lexicon = len(self.unigramcounts)-1

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        """Edge case: w appear at the beginning of sentence"""
        if trigram[:2] == ('START', 'START'):
            return self.trigramcounts[trigram] / self.number_of_sentence

        """If context u, v has not been seen before"""
        if self.bigramcounts[trigram[:2]] == 0:
            return 1/self.size_of_lexicon

        return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        return self.unigramcounts[unigram]/self.total_word_counts

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return (lambda1 * self.raw_trigram_probability(trigram)
                + lambda2 * self.raw_bigram_probability(trigram[1:3])
                + lambda3 * self.raw_unigram_probability(trigram[2:3]))

        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        ngrams = get_ngrams(sentence,3)

        sentence_logprob = 0
        for ngram in ngrams:
            ngram_probability = math.log2(self.smoothed_trigram_probability(ngram))
            sentence_logprob+=ngram_probability

        return float(sentence_logprob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_sentenceprob = 0
        corpus_total_word_counts = 0

        for sentence in corpus:
            corpus_total_word_counts += 1
            sum_sentenceprob += self.sentence_logprob(sentence)
            for word in sentence:
                corpus_total_word_counts += 1

        l = (1 / corpus_total_word_counts) * sum_sentenceprob

        return float(2**(-l))

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_train_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_train_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total +=1
            if pp_train_high<pp_train_low:
                correct+=1
    
        for f in os.listdir(testdir2):
            pp_train_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_train_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            if pp_train_low<pp_train_high:
                correct += 1
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])
    #print(model.trigramcounts[('START','START','the')])
    #print(model.bigramcounts[('START','the')])
    #print(model.unigramcounts[('the',)])
    #print(model.total_word_counts)
    #print(get_ngrams(["natural","language","processing"],1))




    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    # print(acc)

