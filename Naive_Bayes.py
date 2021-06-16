import utils
from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd 


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


# Remove noise: You will first want to remove noise from your data that is, remove words that don't tell you much about the content. These include all common words like 'I, you, are, is, etc...' that would not give us enough information on the sentiment.
# We'll also remove stock market tickers, retweet symbols, hyperlinks, and hashtags because they can not tell you a lot of information on the sentiment.
# You also want to remove all the punctuation from a tweet. The reason for doing this is because we want to treat words with or without the punctuation as the same word, instead of treating "happy", "happy?", "happy!", "happy," and "happy." as different words.
# Finally you want to use stemming to only keep track of one variation of each word. In other words, we'll treat "motivation", "motivated", and "motivate" similarly by grouping them within the same stem of "motiv-".

custom_tweet = 'RT@ Twitter @chapagain Hello There! Have a great day, :) #good #morning http://chapagain.com.np'
print(process_tweet(custom_tweet))


# Create a function count_tweets that takes a list of tweets as input, cleans all of them, and returns a dictionary.

# The key in the dictionary is a tuple containing the stemmed word and its class label, e.g. ("happi",1).
# The value the number of times this word appears in the given collection of tweets (an integer).

def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''

   
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)

                # if the key exists in the di1tionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
   

    return result

# Testing your function

result = {}
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
print(count_tweets(result, tweets, ys))

# Naive bayes is an algorithm that could be used for sentiment analysis. It takes a short time to train and also has a short prediction time.

# So how do you train a Naive Bayes classifier?
# The first part of training a naive bayes classifier is to identify the number of classes that you have.
# You will create a probability for each class.  P(Dpos)  is the probability that the document is positive.  P(Dneg)  is the probability that the document is negative. Use the formulas as follows and store the values in a dictionary:
# P(Dpos)=DposD(1)
# P(Dneg)=DnegD(2)
# Where  D  is the total number of documents, or tweets in this case,  Dpos  is the total number of positive tweets and  Dneg  is the total number of negative tweets.

# Prior and Logprior
# The prior probability represents the underlying probability in the target population that a tweet is positive versus negative. In other words, if we had no specific information and blindly picked a tweet out of the population set, what is the probability that it will be positive versus that it will be negative? That is the "prior".

# The prior is the ratio of the probabilities P(Dpos)P(Dneg). We can take the log of the prior to rescale it, and we'll call this the logprior

# logprior=log(P(Dpos)P(Dneg))=log(DposDneg)
# .

# Note that log(AB) is the same as log(A)−log(B). So the logprior can also be calculated as the difference between two logs:

# logprior=log(P(Dpos))−log(P(Dneg))=log(Dpos)−log(Dneg)(3)
# Positive and Negative Probability of a Word
# To compute the positive probability and the negative probability for a specific word in the vocabulary, we'll use the following inputs:

# freqpos and freqneg are the frequencies of that specific word in the positive or negative class. In other words, the positive frequency of a word is the number of times the word is counted with the label of 1.
# Npos and Nneg are the total number of positive and negative words for all documents (for all tweets), respectively.
# V is the number of unique words in the entire set of documents, for all classes, whether positive or negative.
# We'll use these to compute the positive and negative probability for a specific word using this formula:

# P(Wpos)=freqpos+1Npos+V(4)
# P(Wneg)=freqneg+1Nneg+V(5)
# Notice that we add the "+1" in the numerator for additive smoothing. This wiki article explains more about additive smoothing.

# Log likelihood
# To compute the loglikelihood of that very same word, we can implement the following equations:

# loglikelihood=log(P(Wpos)P(Wneg))(6)
# Create freqs dictionary
# Given your count_tweets() function, you can compute a dictionary called freqs that contains all the frequencies.
# In this freqs dictionary, the key is the tuple (word, label)
# The value is the number of times it has appeared.

freqs = count_tweets({}, train_x, train_y)

# Given a freqs dictionary, train_x (a list of tweets) and a train_y (a list of labels for each tweet), implement a naive bayes classifier.

# Calculate  V 
# You can then compute the number of unique words that appear in the freqs dictionary to get your  V  (you can use the set function).
# Calculate  freqpos  and  freqneg 
# Using your freqs dictionary, you can compute the positive and negative frequency of each word  freqpos  and  freqneg .
# Calculate  Npos  and  Nneg 
# Using freqs dictionary, you can also compute the total number of positive words and total number of negative words  Npos  and  Nneg .
# Calculate  D ,  Dpos ,  Dneg 
# Using the train_y input list of labels, calculate the number of documents (tweets)  D , as well as the number of positive documents (tweets)  Dpos  and number of negative documents (tweets)  Dneg .
# Calculate the probability that a document (tweet) is positive  P(Dpos) , and the probability that a document (tweet) is negative  P(Dneg) 
# Calculate the logprior
# the logprior is  log(Dpos)−log(Dneg) 
# Calculate log likelihood
# Finally, you can iterate over each word in the vocabulary, use your lookup function to get the positive frequencies,  freqpos , and the negative frequencies,  freqneg , for that specific word.
# Compute the positive probability of each word  P(Wpos) , negative probability of each word  P(Wneg)  using equations 4 & 5.
# P(Wpos)=freqpos+1Npos+V(4)
# P(Wneg)=freqneg+1Nneg+V(5)
# Note: We'll use a dictionary to store the log likelihoods for each word. The key is the word, the value is the log likelihood of that word).

# You can then compute the loglikelihood:log(P(Wpos)P(Wneg))(6)

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

 

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_x)

    # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    D_pos = sum([1 for i in train_y if i==1])

    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = sum([1 for i in train_y if i==0])

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word,1),0)
        freq_neg = freqs.get((word,0),0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos)-np.log(p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))


# naive bayes
# Now that we have the logprior and loglikelihood, we can test the naive bayes function by making predicting on some tweets!

# Implement naive_bayes_predict
# Instructions: Implement the naive_bayes_predict function to make predictions on tweets.

# The function takes in the tweet, logprior, loglikelihood.
# It returns the probability that the tweet belongs to the positive or negative class.
# For each tweet, sum up loglikelihoods of each word in the tweet.
# Also add the logprior to this sum to get the predicted sentiment of that tweet.
# p=logprior+∑iN(loglikelihoodi)
 
# Note
# Note we calculate the prior from the training data, and that the training data is evenly split between positive and negative labels (4000 positive and 4000 negative tweets). This means that the ratio of positive to negative 1, and the logprior is 0.

# The value of 0.0 means that when we add the logprior to the log likelihood, we're just adding zero to the log likelihood. However, please remember to include the logprior, because whenever the data is not perfectly balanced, the logprior will be a non-zero value

def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''

    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p

my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

# implement test_naive_bayes
# Instructions:

# Implement test_naive_bayes to check the accuracy of your predictions.
# The function takes in your test_x, test_y, log_prior, and loglikelihood
# It returns the accuracy of your model.
# First, use naive_bayes_predict function to make predictions for each tweet in text_x.

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly

    
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = sum(abs(y_hats - test_y))/len(test_y)

    # Accuracy is 1 minus the error
    accuracy = 1 - error

    return accuracy

print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))


for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
#     print(f'{tweet} -> {p:.2f} ({p_category})')
    print(f'{tweet} -> {p:.2f}')

my_tweet = 'you are bad :('
naive_bayes_predict(my_tweet, logprior, loglikelihood)


#  Filter words by Ratio of positive to negative counts
# Some words have more positive counts than others, and can be considered "more positive". Likewise, some words can be considered more negative than others.
# One way for us to define the level of positiveness or negativeness, without calculating the log likelihood, is to compare the positive to negative frequency of the word.
# Note that we can also use the log likelihood calculations to compare relative positivity or negativity of words.
# We can calculate the ratio of positive to negative frequencies of a word.
# Once we're able to calculate these ratios, we can also filter a subset of words that have a minimum ratio of positivity / negativity or higher.
# Similarly, we can also filter a subset of words that have a maximum ratio of positivity / negativity or lower (words that are at least as negative, or even more negative than a given threshold).
# Implement get_ratio()
# Given the freqs dictionary of words and a particular word, use lookup(freqs,word,1) to get the positive count of the word.
# Similarly, use the lookup() function to get the negative count of that word.
# Calculate the ratio of positive divided by negative counts
# 𝑟𝑎𝑡𝑖𝑜=pos_words+1neg_words+1
 
# Where pos_words and neg_words correspond to the frequency of the words in their respective classes.

def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words
        word: string to lookup

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
   
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs,word,1)

    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs,word,0)

    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)

    return pos_neg_ratio

print(get_ratio(freqs, 'happi'))

# Implement get_words_by_threshold(freqs,label,threshold)
# If we set the label to 1, then we'll look for all words whose threshold of positive/negative is at least as high as that threshold, or higher.
# If we set the label to 0, then we'll look for all words whose threshold of positive/negative is at most as low as the given threshold, or lower.
# Use the get_ratio() function to get a dictionary containing the positive count, negative count, and the ratio of positive to negative counts.
# Append a dictionary to a list, where the key is the word, and the dictionary is the dictionary pos_neg_ratio that is returned by the get_ratio() function. An example key-value pair would have this structure:
# {'happi':
#   {'positive': 10, 'negative': 20, 'ratio': 0.524}
# }

def get_words_by_threshold(freqs, label, threshold):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio


        # otherwise, do not include this word in the list (do nothing)
    return word_list
    
# Test your function: find negative words at or below a threshold
print(get_words_by_threshold(freqs, label=0, threshold=0.05))

# Test your function; find positive words at or above a threshold
get_words_by_threshold(freqs, label=1, threshold=10)


# Test with your own tweet - feel free to modify `my_tweet`
my_tweet = 'I am happy because I am learning :)'

p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print(p)


# Some error analysi
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))