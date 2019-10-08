# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np
import random 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
lemmatizer = WordNetLemmatizer() 
class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        pss = ProbabilisticSequenceScorer(self.tag_indexer,self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        pred_tags = []
        num_tags = len(self.tag_indexer)
        num_words = len(sentence_tokens)
        viterbi = np.zeros((num_words,num_tags))
        backpointer = np.zeros((num_words,num_tags))

        #Initialization
        for tag_idx in range(num_tags):
            word_index = self.word_indexer.index_of(sentence_tokens[0].word)
            viterbi[0][tag_idx] = self.init_log_probs[tag_idx] + pss.score_emission(sentence_tokens,tag_idx,0)
            backpointer[0][tag_idx] = 0

        #Recursion
        for word_idx in range(1, num_words):
            for tag_idx in range(num_tags):
                word_index = self.word_indexer.index_of(sentence_tokens[word_idx].word)
                yprev_max = np.zeros(num_tags)
                for it in range(0,num_tags):
                    yprev_max[it] = self.transition_log_probs[it][tag_idx] + viterbi[word_idx-1][it]               
                    #nxT matrix, n = no. of words, T = no. of tags
                viterbi[word_idx][tag_idx] = pss.score_emission(sentence_tokens,tag_idx,word_idx) + np.max(yprev_max)
                backpointer[word_idx][tag_idx] = np.argmax(yprev_max)
        
        #Termination
        idx = np.argmax(viterbi[-1,:])
        pred_tags.append(self.tag_indexer.get_object(idx))
        previous = idx
        for t in range(num_words-1,0,-1):
            pred_tags.insert(0,self.tag_indexer.get_object(backpointer[t][int(idx)]))
            idx = backpointer[t][int(idx)]
            
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i]) #e.g. tag ['B-LOC', 'I-LOC', 'O'] e.g. sentence ['Token(Gloria, NNP, I-NP)', 'Token(Bistrita, NNP, I-NP)']
            # ['(4, 5, LOC)', '(6, 8, ORG)'], the fourth word is a location, 6th&7th word in ORG
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    print(transition_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print(transition_counts)
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class FeatureBasedSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
    """
    def __init__(self, tag_indexer: Indexer, transition_log_probs: np.ndarray, feature_indexer, feature_weights, feature_cache):
        self.tag_indexer = tag_indexer
        self.transition_log_probs = transition_log_probs
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.feature_cache = feature_cache

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx][curr_tag_idx]

    def score_emission(self, word_idx: int, tag_idx: int):
        return score_indexed_features(self.feature_cache[word_idx][tag_idx], self.feature_weights)

class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_log_probs, actual_sentences, tf_idf_score, feature_names, words_to_tag_counters):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_log_probs = transition_log_probs
        self.actual_sentences = actual_sentences
        self.tf_idf_score = tf_idf_score
        self.feature_names = feature_names
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence_tokens):
        # 4-d list indexed by sentence index, word index, tag index, feature index
        pred_tags = []
        num_tags = len(self.tag_indexer)
        num_words = len(sentence_tokens)

        ##Counting words/tag and their count
        words_to_tag_counters = {}
        for idx in range(0, len(sentence_tokens)):
            word = sentence_tokens[idx].word
            if not word in words_to_tag_counters:
                words_to_tag_counters[word] = 1
            else:
                words_to_tag_counters[word] += 1

        sent = self.actual_sentences
        words = []
        for token_idx in range(len(sentence_tokens)):
            words.append(sentence_tokens[token_idx].word)
        sent.append(words)
        self.actual_sentences = [' '.join(s) for s in sent]

        test_set_feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence_tokens))]
        
        for tag_idx in range(num_tags):
            cur_tag = self.tag_indexer.get_object(tag_idx)
            for prev_tag_idx in range(num_tags):
                prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                if prev_tag[0] == 'O' and cur_tag[0] == 'I':
                    self.transition_log_probs[prev_tag_idx][tag_idx] = -np.inf
                elif cur_tag[0] == 'I':
                    if prev_tag[2:] != cur_tag[2:]:
                        self.transition_log_probs[prev_tag_idx][tag_idx] = -np.inf
        
        tf_idf_score = 0
        #Add current state to the training set of sentences for TF-IDF computation
        for word_idx in range(0, len(sentence_tokens)):
            cur_word = sentence_tokens[word_idx].word.lower().translate(str.maketrans('', '', string.punctuation))
            if cur_word in self.feature_names:
                tf_idf_score = max(self.tf_idf_score[cur_word])#sentence_idx]
            else:
                tf_idf_score = 0
            for tag_idx in range(0, len(self.tag_indexer)):
                test_set_feature_cache[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, self.words_to_tag_counters, tf_idf_score, add_to_indexer=False)
        
        fss = FeatureBasedSequenceScorer(self.tag_indexer,self.transition_log_probs, self.feature_indexer, self.feature_weights, test_set_feature_cache)
        
        viterbi = np.zeros((num_words,num_tags))
        backpointer = np.zeros((num_words,num_tags))

        #Inititalization
        for tag_idx in range(num_tags):
            viterbi[0][tag_idx] = fss.score_emission(0,tag_idx)
        
        #Recursion
        for word_idx in range(1,num_words):
            for tag_idx in range(num_tags):
                yprev = np.zeros(num_tags)
                for it in range(0,num_tags):
                    yprev[it] = fss.score_transition(it,tag_idx) + viterbi[word_idx-1][it]               
                #nxT matrix, n = no. of words, T = no. of tags
                viterbi[word_idx][tag_idx] = fss.score_emission(word_idx, tag_idx) + np.max(yprev)
                backpointer[word_idx][tag_idx] = np.argmax(yprev)
        
        #Termination step
        bp = np.argmax(viterbi[-1,:])
        for t in reversed(range(num_words)):
            pred_tags.append(self.tag_indexer.get_object(bp))
            bp = backpointer[t][int(bp)]
                 
        pred_tags.reverse()
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")


    #Counting words/tag and their count
    words_to_tag_counters = {}
    for sentence in sentences:
        tags = sentence.get_bio_tags()
        for idx in range(0, len(sentence)):
            word = sentence.tokens[idx].word
            if not word in words_to_tag_counters:
                words_to_tag_counters[word] = 1
            else:
                words_to_tag_counters[word] += 1

    #TF - IDF
    tfidf = TfidfVectorizer()
    sent = []
    for sentence in sentences:
        words = []
        for idx in range(0, len(sentence)):
            words.append(lemmatizer.lemmatize(sentence.tokens[idx].word))
        sent.append(words)
    actual_sentences = [' '.join(s) for s in sent]
    X = tfidf.fit_transform(actual_sentences)
    df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names())
    feature_names = tfidf.get_feature_names()


    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    stop_words = set(stopwords.words('english'))
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            cur_word = sentences[sentence_idx].tokens[word_idx].word#lower().translate(str.maketrans('', '', string.punctuation))#curr_word.tolower())
            if cur_word in feature_names:
                tf_idf_score = df[cur_word][sentence_idx]
            else:
                tf_idf_score = 0
            for tag_idx in range(0, len(tag_indexer)):
                if sentences[sentence_idx].tokens[word_idx].word not in stop_words: 
                    feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, words_to_tag_counters, tf_idf_score, add_to_indexer=True)
    print("Training")
    

    #getting the transition log probabilities from the HMM model
    hmm_model = train_hmm_model(sentences)
    transition_log_probs = hmm_model.transition_log_probs    

    num_tags = len(tag_indexer)
    num_sent = len(sentences)
    num_features = len(feature_indexer)
    feature_weights = [random.random() for i in range(num_features)]
    num_epochs = 3
    gradient_ascent = UnregularizedAdagradTrainer(feature_weights)
    for epoch in range(num_epochs):
        for sentence_idx in range(num_sent):
            if sentence_idx % 100 == 0:
                print("Ex %i/%i" % (sentence_idx, num_sent))
            num_words = len(sentences[sentence_idx])
            alpha = np.zeros((num_words, num_tags))
            beta = np.zeros((num_words, num_tags))

            #FB initialization
            for tag_idx in range(num_tags):
                    alpha[0][tag_idx] = gradient_ascent.score(feature_cache[sentence_idx][0][tag_idx])
                    beta[num_words-1][tag_idx] = 0
            
            #forward algorithm
            for word_idx in range(1,num_words):
                for cur_tag in range(num_tags):
                    emission_log_prob = gradient_ascent.score(feature_cache[sentence_idx][word_idx][cur_tag])                        
                    for prev_tag in range(num_tags):
                        alpha_val = alpha[word_idx-1][prev_tag] + emission_log_prob
                        if prev_tag ==0:
                            alpha[word_idx][cur_tag] = alpha_val
                        else:
                            alpha[word_idx][cur_tag] = np.logaddexp(alpha[word_idx][cur_tag], alpha_val)
                    
            #backward algorithm
            for word_idx in range(num_words-2,0,-1):
                for cur_tag in range(num_tags):
                    for next_tag in range(num_tags):
                        emission_log_prob = gradient_ascent.score(feature_cache[sentence_idx][word_idx+1][next_tag])
                        beta_val = beta[word_idx+1][next_tag] + emission_log_prob
                        if prev_tag == 0:
                            beta[word_idx][tag_idx] = beta_val
                        else:
                            beta[word_idx][tag_idx] = np.logaddexp(beta[word_idx][cur_tag], beta_val)

            #computing marginal probabilities
            denominator = np.zeros(num_words)
            for word_idx in range(num_words):
                denominator[word_idx] = alpha[word_idx][0] + beta[word_idx][0]
                for tag_idx in range(1, num_tags):
                    val = alpha[word_idx][tag_idx] + beta[word_idx][tag_idx]
                    denominator[word_idx] = np.logaddexp(val, denominator[word_idx])

            marginal_prob = np.zeros((num_words, num_tags)) 
            for word_idx in range(num_words):
                for tag_idx in range(num_tags):
                    marginal_prob[word_idx][tag_idx] = (alpha[word_idx][tag_idx] + beta[word_idx][tag_idx] ) - denominator[word_idx]

            #computing the gradient
            grad_count = Counter()
            for word_idx in range(num_words):
                gold_label = sentences[sentence_idx].get_bio_tags()[word_idx]
                gold_label_idx = tag_indexer.index_of(gold_label)
                for feature in feature_cache[sentence_idx][word_idx][gold_label_idx]:
                    grad_count[feature] += 1
                for tag_idx in range(num_tags):
                    for feature in feature_cache[sentence_idx][word_idx][tag_idx]:
                        grad_count[feature] += -np.exp(marginal_prob[word_idx][tag_idx])
            gradient_ascent.apply_gradient_update(grad_count,1)
    
    crfmodel = CrfNerModel(tag_indexer, feature_indexer, gradient_ascent.weights, transition_log_probs, actual_sentences, df, feature_names, words_to_tag_counters)
    np.save("feature_weights", gradient_ascent.weights)
    """
    weights = np.load('feature_weights.npy')
    crfmodel = CrfNerModel(tag_indexer, feature_indexer, weights, transition_log_probs, actual_sentences, df, feature_names, words_to_tag_counters)
    """
    return crfmodel

def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, words_to_tag_counters, tf_idf_score, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    # curr_word = lemmatizer.lemmatize(sentence_tokens[word_index].word)
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordCount=" + repr(words_to_tag_counters[curr_word]))
    
    if tf_idf_score >= 0.75:
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TF-IDF=" +"1-TFIDF")
    elif tf_idf_score >= 0.5:
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TF-IDF=" +"0.75-TFIDF")
    elif tf_idf_score >= 0.25:
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TF-IDF=" +"0.5-TFIDF")
    else:
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TF-IDF=" +"0.25-TFIDF")
    return np.asarray(feats, dtype=int)

