# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:28:46 2021

@author: Asus
"""

### PART 1: TEXT MINING

# We will start by importing the requested libraries.
import pandas as pd # for vectorised operations.
import numpy as np # for vectorised operations.
from nltk.corpus import stopwords # to remove the stopwords.
import matplotlib.pyplot as plt # to plot the histogram in question 3.
import time # to calculate the execution time.
from sklearn.preprocessing import OrdinalEncoder # so we can encode the labels.
import re # for the text processing in question 4.
from sklearn.feature_extraction.text import CountVectorizer # to create a sparse representation of the term-document matrix.
from sklearn.naive_bayes import MultinomialNB # to perform a Multinomial Naive Bayes Classification model.
from sklearn import metrics # To evaluate our model.

initial_start_time = time.time() # here we store the time at which the program starts.

# Now we download and save the Coronavirus Tweets NLP dataframe in an object which we will call 'df'. We
# have to pass a specific type of encoding ('latin-1') so pandas can interpret the tweets correctly.
df = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')



# 1. [20 points] Compute the possible sentiments that a tweet may have, the second most popular
# sentiment in the tweets, and the date with the greatest number of extremely positive tweets.
# Next, convert the messages to lower case, replace non-alphabetical characters with whitespaces
# and ensure that the words of a message are separated by a single whitespace.


start_time = time.time() # to measure the execution time of this question.

# We print the order of the questions so it is easier to follow through the terminal.
print("\nFIRST QUESTION\n")

# 1.1: Possible sentiments a tweet may have:
print("The possible sentiments a tweet may have are the following: {}.\n".format(df.Sentiment.unique()))


# 1.2: Second most popular sentiment in the tweets:
print("The second most popular sentiment in the tweets is '{}', which has a total of {} tweets.\n".format(df.Sentiment.value_counts().index[1], 
                                                                                                          df. Sentiment.value_counts()[1]))

# 1.3: Date with the greatest number of extremely positive tweets:

# We first convert the strings in the TweetAt column into a timestamp format.
df['TweetAt'] = pd.to_datetime(df['TweetAt'], format='%d-%m-%Y')

# Now we print the day with the most extremely positive tweets and the number of extremely tweets made
# that day.
print("The day with the greatest number of extremely positive tweets was the '{}', which had a total of {} extremely positive tweets.\n".format(df[df['Sentiment']=='Extremely Positive'].groupby('TweetAt')['Sentiment'].count().sort_values(ascending=False).index[0].date(),
                                                                                                                                              df[df['Sentiment']=='Extremely Positive'].groupby('TweetAt')['Sentiment'].count().sort_values(ascending=False)[0]))

# 1.4: Convert the tweets in the dataframe to ease the later analysis:

    # 1.4.1: Convert the messages to lower case: We apply the .lower() function to all the texts in our
    # dataframe.
df['ModifiedTweet'] = df['OriginalTweet'].apply(str.lower)

    # 1.4.2: Replace non-alphabetical characters with whitespaces:
        # First, we create a function (alpha) that accepts as an input a message and returns that same
        # message with the non-alhpabetical characters removed and replaced with a whitespace.
def alpha(msg):
    for x in msg:
        if not x.isalpha():
            msg = str.replace(msg, x, ' ')
    return msg

    # Now we apply the alpha function that replaces the non-alphabetical characters with a whitespace
    # to all the texts in our dataframe through the .apply() function.
df['ModifiedTweet'] = df['ModifiedTweet'].apply(alpha)

    # 1.4.3: Ensure that the words of a message are separated by a single whitespace: we apply the split
    # and join function to all the texts in our dataframe to ensure there is only one whitespace between
    # each word.
df['ModifiedTweet'] = df['ModifiedTweet'].apply(lambda x: " ".join(x.split()))

print("Execution time for question 1: {:.3f} seconds\n".format(time.time()-start_time))



# 2. [20 points] Tokenize the tweets (i.e. convert each into a list of words), count the total number
# of all words (including repetitions), the number of all distinct words and the 10 most frequent
# words in the corpus. Remove stop words, words with less than 3 characters and recalculate the number
# of all words (including repetitions) and the 10 most frequent words in the modified corpus. What
# do you observe?


start_time = time.time() # to measure the execution time of this question.

print("\nSECOND QUESTION\n")

# 2.1: Tokenize the tweets: we create a new column called 'TokenizedTweet', which will store a list of the 
# words in each 'ModifiedTweet'.
df['TokenizedTweet'] = df['ModifiedTweet'].apply(str.split)


# 2.2: Count the total number of all words (including repetitions): for this exercise we will create a dictionary
# that will contain ALL the words in the document as a key. This dictionary will have key-value pairs as word-number
# of ocurrences of that word. This will have to be done sooner or later, so we better do it now to make our program 
# more efficient and save performance time.

# First, we create the corpus (list of all the words in our tweets).
corpus =  []
for i in range(len(df)):
    corpus += df.TokenizedTweet[i] # We store all the words in this corpus

# In the dictionary we will store the words as a key, and the number of ocurrences of each word as the value
# of the corresponding key.
words = {} # we initialise the dictionary

# Now we have to loop through all the corpus and create a key if it does not exist yet, and in case it 
# exists, add 1 to its value
for word in corpus:
    if word not in words.keys(): # If the word is not in the dictionary yet...
        words[word] = 1 # we create the key-value item and initalise the value to 1 (first appearance).
    else: # If the word is included already in the dictionary...
        words[word] += 1 # we add 1 to the current value.

# Now we can print the number of words (repetitions included) in our tweets dataframe. This will be the length
# of the corpus, or the sum of all the values in our dictionary.
print("The total number of words (repetitiones included) in the whole dataset is: {}.\n".format(len(corpus)))


# 2.3: Count the number of all distinct words: we will benefit from the dictionary created in the above point to 
# print the number of keys in the 'words' dictionary (each key is a unique word in the tweets corpus). We could
# also convert the corpus list into a set, which does not accept duplicates, and print its length.
print("The total number of unique words in the whole dataset is: {}.\n".format(len(words.keys())))


# 2.4: Count the 10 most frequent words in the corpus: for this exercise we will also use the dictionary 'words'. We
# have to sort it out in descending order respect to the values, and print the 10 most frequent words.
print("The 10 most frequent words of the tweets corpus are the following:\n")

# We will print the most frequent words directly:
for i in range(10): # we print the most frequent words in order
   print("{}. {}, {} occurrences".format(i+1, sorted(words.items(), key=lambda x: x[1], reverse=True)[i][0], 
                                         sorted(words.items(), key=lambda x: x[1], reverse=True)[i][1]))


# 2.5: Remove stopwords and words with less than three characters: we will benefit from the 'corpus' list. We
# are going to create a second dictionary ('words2') that will include words in the dictionary only if it is not
# a stopword and if it has more than 2 characters.

# First we store the stopwords in the list 'sw'. The stopwords are obtained from the library NLTK.
sw = stopwords.words('english')

# Now we create the new dictionary without these uninformative words. We also create a set, 'discarded', where we will
# store these useless words.
words2 = {} # we initialise the new dictionary without stopwords and short words.
discarded = set() # here we will store the words that we are removed from our vocabulary to benefit from it later.

# Now we add elements to our dictionary and to our set of discarded words.
for word in corpus: # loop through each word in our corpus,
    if word not in sw: # if the word is not a stopword,
        if len(word)>2: # and if it has more than 2 characters:
            if word not in words2.keys(): # If the word is not in the new dictionary yet...
                words2[word] = 1 # we create the key-value item and initalise the value to 1 (first appearance).
            else: # If the word is included already in the dictionary...
                words2[word] += 1 # we add 1 to the current value.
        else: # if a word from the corpus is a short word (2 or less characters).
            discarded.add(word)
    else: # if a word from the corpus appears in our stopwords.
        discarded.add(word)

                
# 2.6: Recalculate the number of all words (including repetitions): we just have to repeat the same thing we did
# in 2.4 but now using the modified dictionary ('words2').

# Now we can also print the number of words (repetitions included) in our tweets dataframe after removing the 
# stopwords. This will be the length of the modified corpus, or the sum of all the values in our second dictionary.
print("\nThe total number of words (repetitiones included) after removing stopwords is: {}.\n".format(sum(words2.values())))

# We will benefit from the dictionary created in the above point to print the number of keys in the 'words2' dictionary
# (each key is a unique word in the tweets corpus). This will be the number of words in our corpus after removing short
# words and stopwords:
print("The total number of unique words after removing stopwords is: {}.\n".format(len(words2.keys())))

                
# 2.6: Recalculate the 10 most frequent words in the modified corpus: we just have to repeat the same thing we did
# in 2.4 but now using the modified dictionary ('words2').
print("The 10 most frequent words of the tweets corpus (excluding stopwords and very short words, 2 characters or less) are the following:\n")

# We will print the most frequent words directly:
for i in range(10): # we print the most frequent words in order
   print("{}. {}, {} occurrences".format(i+1, sorted(words2.items(), key=lambda x: x[1], reverse=True)[i][0], sorted(words2.items(), key=lambda x: x[1], reverse=True)[i][1]))

print("\nExecution time for question 2: {:.3f} seconds\n".format(time.time()-start_time))



# 3. [10 points] Plot a histogram with word frequencies, where the horizontal axis corresponds to
# words, while the vertical axis indicates the fraction of documents in which a word appears. The
# words should be sorted in increasing order of their frequencies. Because the size of the data
# set is quite big, use a line chart for this, instead of a histogram. In what way this plot can be
# useful for deciding the size of the term document matrix? How many terms would you add in a
# term-document matrix for this data set?


start_time = time.time() # to measure the execution time of this question.

print("\nTHIRD QUESTION\n")

# First, we need to create some kind of counter that contains the number of documents a word appears in.
# To do so, I will create a pandas Series. The next step is a bit tricky, so I will explain it step by step.
# The final result is a pandas Series which I have called 'frequencies'
 
    # 1. As we are only interested in the words that APPEAR on the tweet, we can convert the 'TokenizedTweet'
    # column in a set, which removes duplicates.
step1 = df.TokenizedTweet.apply(set)

    # 2. We currently have the unique words that appear in each tweet. Next step will be to create a dataframe 
    # with the words that appear in each tweet. We do this because we are going to .stack() them later, I will
    # explain this step next. In order to create a DataFrame I have to pass a list to the pd.DataFrame method.
    # If not, it will create a DataFrame with sets as values. That is why I use .to_list().
step2 = pd.DataFrame(step1.to_list())
 
    # 3. We currently have a dataframe that contains 1 row per tweet. And in each row we have a cell for each
    # unique word that appears in the tweet (no duplicate values). Now we can apply the .stack() method mentioned
    # in step 2. This method literally stacks the elements in the rows and makes a multi-index. The first index
    # will be the tweet number (going from 0 to len(df)-1) and the inner index will go from 0 to 
    # len(set(TokenizedTweet))-1, meaning that each inner index will have a different number of values depending
    # on the length of each tweet. This actually does not manner, what matters is that we have as values the unique
    # words that appear in each tweet.
step3 = step2.stack()

    # 4. We currently have a multi-index Series of the unique words that appear in each tweet as values. We have to
    # apply value_counts(), which will return the number of documents a word appears in. This will return a pandas
    # Series where the index will contain the unique words, and the values are the number of documents each word
    # appears in.
step4 = step3.value_counts()
   
    # 5. We currently have a Series where the index is composed by ALL the unique words, including the stopwords
    # and the words with 2 characters or less. We are not interested in these words, so we will drop them from 
    # our Series by applying the .drop() method. Here is where we will use the 'discarded' set we created in the
    # step 2.5. We only have to pass that variable as a parameter in the .drop() method.
step5 = step4.drop(labels=discarded)

    # 6. We currently have a Series of len(words2.keys()), meaning we have the unique words after removing the 
    # stopwords and the very short words. As the question asks us to plot the frequencies, we have to divide all
    # the values in our Series by the total number of documents in our corpus (len(df)).
    # Final result:
frequencies = step5/len(df)    

# After all this, we have a Series with the index as the values we want to put in the horizontal index, and with the
# values as the values we want to plot in the vertical axis. So we will use Matplotlib to plot them. The good thing is 
# that they are already sorted as the value_counts returns the Series in descending order, we only have to invert
# the Series so it is plotted in increased order.
fig = plt.figure(figsize=(15, 18)) # set the figure size.
ax1 = fig.add_subplot(211) # create first subplot, the one on top.
ax1.plot(frequencies.sort_values().index, frequencies.sort_values().values) # create the lineplot.
ax1.set_xticks(np.arange(1, len(frequencies), 2500)) # add only some words (one of 2500) to the plot.
ax1.tick_params(axis="x", rotation=90) # rotate the x-axis ticks (the words of our vocabulary).
ax1.set(xlabel='WORDS', ylabel='FREQUENCIES') # set the axis titles.
ax1.set_title("COMPLETE VOCABULARY", fontweight='bold') # set the subplot title.

# As this plot does not give us much information, I will also plot the top 20 words so we can have a sense of
# how the curve evolves
ax2 = fig.add_subplot(212) # create second plot, the one below.
ax2.plot(frequencies.sort_values().index[len(frequencies)-20:], frequencies.sort_values().values[len(frequencies)-20:]) # create the lineplot.
ax2.tick_params(axis="x", rotation=90) # rotate the x-axis ticks (top-20 words of our vocabulary).
ax2.set(xlabel='WORDS', ylabel='FREQUENCIES') # set the axis titles.
ax2.set_title("TOP-20", fontweight='bold') # set the subplot title.

print('Plotting the histogram...\n')
plt.tight_layout() # so our plot is not so crushed.
plt.show() # render the plots.

print("Execution time for question 3: {:.3f} seconds\n".format(time.time()-start_time))



# 4. [10 points] This task can be done individually from the previous three. Produce a Multinomial
# Naive Bayes classifier for the Coronavirus Tweets NLP data set using scikit-learn. For this, store
# the corpus in a numpy array, produce a sparse representation of the term-document matrix with
# a CountVectorizer and build the model using this term-document matrix. What is the error rate
# of the classifier? You may want to check the scikit-learn documentation for performing this task.


start_time = time.time() # to measure the execution time of this question.

print("\nFOURTH QUESTION\n")

# As we have the data and the labels, we have a supervised learning problem. We will store the labels
# in a y array, after that we will encode the labels from strings to numbers, to ease the execution.
y = df.Sentiment.values # we save the target array in y.

# Now we set the numerical attribute we want to encode our categorical attributes. 
oe = OrdinalEncoder(categories=[np.array(['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])])
y = oe.fit_transform(y.reshape(-1, 1)).flatten() # we fit and transform our target attribute and we flatten it.

# Now we have to clean and prepare the data we will use to train our model. We will start from scratch, this
# way we will remove stopwords and urls too.

# We will start by saving the raw data (tweets) in a new dataframe, 'x'.
x = pd.DataFrame(df['OriginalTweet'])
# Now, we will convert all the tweets in lowercase
x['ModTweet'] = x['OriginalTweet'].apply(str.lower)
# Next, we will remove links and URLs from our tweets. We will do so by helping ourselves with the RegEx library.
x['ModTweet'] = x['ModTweet'].apply(lambda x: re.sub(r'http\S+', '', x))
# We will also remove usernames from the tweets, as they do not give much information
x['ModTweet'] = x['ModTweet'].apply(lambda x: re.sub(r'@\S+', '', x))
# Lastly, we will replace non-alphabetical characters with whitespaces and ensure that the words of a message
# are separated by a single whitespace. We will benefit from the 'alpha' function created above.
x['ModTweet'] = x['ModTweet'].apply(alpha) # remove non-alphabetical characters.
x['ModTweet'] = x['ModTweet'].apply(lambda x: " ".join(x.split())) # ensure there is only one whitespace between words.

# Next step is to produce a sparse representation of the term-document matrix with a CountVectorizer object. As
# we have removed a lot of words and characters that do no add much value to the analysis, our corpus is somewhat
# clean already. However, the last step would be to remove the stopwords too. This can be done directly when
# creating the CountVectorizer object:

# We create the CountVectorizer object and include all the words in our tweets dataset.
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
# We apply it to the tweets in our dataframe (x.ModTweet)
X = vectorizer.fit_transform(np.array(x.ModTweet))

# Now we can train and evaluate our model on the data we have. As this is a complex task, we will not divide
# our data in train and test. We will train our model with all the data we have at our hand. Later, we will
# evaluate on all the train set. This way, our model can learn better.

# We will start by initialising our model object (Multinomial Naive Bayes classifier).
nb = MultinomialNB(alpha=1e-6) # our model will be called 'nb'.

# Now we train our model with the data we have at hand:
nb.fit(X, y)

# And lastly, we evaluate our model by predicting the target of the data instances we have and by comparing
# it to the actual targets stored in 'y'
y_pred = nb.predict(X) # first we predict,
print("The training accuracy of the model is {:.3f}%.\n".format(metrics.accuracy_score(y, y_pred)*100)) # then we evaluate.

print("Execution time for question 4: {:.3f} seconds\n".format(time.time()-start_time))

print("\nTOTAL EXECUTION TIME: {:.3f} seconds".format(time.time()-initial_start_time))