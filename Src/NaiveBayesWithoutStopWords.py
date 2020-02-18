from __future__ import division
import glob
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#Naive bayes after filtering out stop words
def NaiveBayesWithoutStopWords():

    no_of_ham_files = len(glob.glob("train/ham/*.txt"))
    no_of_spam_files = len(glob.glob("train/spam/*.txt"))
    total_no_of_files = no_of_ham_files + no_of_spam_files

    #Using tokenizer to break sentences into tokens
    tokenizer = RegexpTokenizer("[a-zA-Z]+")

    #Using SnowballStemmer as given in the documentation of assignment to get root words
    stemmer = SnowballStemmer("english")

    #Using nltk to filter out stopwords
    stop_words = set(stopwords.words("english"))

    total_no_of_words_in_whole_train_folder = 0
    bag_of_words = {}
    path_of_all_training_files = glob.glob("train/*/*" + ".txt")

    #Reading all the files inside train folder and gathering all the words to form the vocabulary
    for file in path_of_all_training_files:
        with open(file, 'r') as fp:
            #Reading each line of the file
            for line in fp:
                #tokenizing
                tokens = tokenizer.tokenize(line)
                #Getting only the unique words means ignoring the plural forms, selecting vocab language as english
                #Filtering out stop words along with stemming
                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_no_of_words_in_whole_train_folder = total_no_of_words_in_whole_train_folder + len(stemmed_words)
                #Once words are stemmed, selecting only the unique words from entire bag of words
                for word in stemmed_words:
                    if word != '':
                        if word.lower() in bag_of_words:
                            bag_of_words[word.lower()] += 1
                        else:
                            bag_of_words[word.lower()] = 1

    #Getting only the unique root words from bag of words 
    no_of_unique_words_in_whole_train_folder = len(bag_of_words)

    path_of_all_ham_files = glob.glob("train/ham/*" + ".txt")
    no_of_unique_ham_words_in_ham_folder = {}
    total_no_of_ham_words_in_ham_folder = 0

    #Reading all the ham files inside train/ham folder and gathering all the ham words
    for file in path_of_all_ham_files:
        with open(file, 'r') as fp:
            #Reading each line of file
            for line in fp:
                #tokenizing
                tokens = tokenizer.tokenize(line)
                #Getting only the unique words means ignoring the plural forms, selecting vocab language as english
                #Filtering out stop words along with stemming
                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_no_of_ham_words_in_ham_folder += len(stemmed_words)
                #Once words are stemmed, selecting only the unique words from entire bag of words
                for word in stemmed_words:
                    if word != '':
                        if word.lower() in no_of_unique_ham_words_in_ham_folder:
                            no_of_unique_ham_words_in_ham_folder[word.lower()] += 1
                        else:
                            no_of_unique_ham_words_in_ham_folder[word.lower()] = 1

    path_of_all_spam_files = glob.glob("train/spam/*" + ".txt")
    no_of_unique_spam_words_in_spam_folder = {}
    total_no_of_spam_words_in_spam_folder = 0

    #Reading all the spam files inside train/spam folder and gathering all the spam words
    for file in path_of_all_spam_files:
        with open(file, 'r') as fp:
            #Reading each line of file
            for line in fp:
                #tokenizing 
                tokens = tokenizer.tokenize(line)
                #Getting only the unique words means ignoring the plural forms, selecting vocab language as english
                #Filtering out stop words along with stemming
                stemmed_words = [stemmer.stem(t) for t in tokens if not t in stop_words]
                total_no_of_spam_words_in_spam_folder += len(stemmed_words)
                #Once words are stemmed, selecting only the unique words from entire bag of words
                for word in stemmed_words:
                    if word != '':
                        if word.lower() in no_of_unique_spam_words_in_spam_folder:
                            no_of_unique_spam_words_in_spam_folder[word.lower()] += 1
                        else:
                            no_of_unique_spam_words_in_spam_folder[word.lower()] = 1

    #Finding probability of each class    
    probability_of_ham_class = math.log((no_of_ham_files / total_no_of_files), 10)
    probability_of_spam_class = math.log10(no_of_spam_files / total_no_of_files)

    #calculating the accuracy of model during test time without stop words
    path_for_ham_test = glob.glob("test/ham/*.txt")
    correct_guesses = 0
    test_file_count = len(path_for_ham_test)

    #Calculating number of correct guesses in test/ham during test time and determining the accuracy without stopwords
    for file in path_for_ham_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r') as fp:
            #Reading each line of the file
            for line in fp:
                #tokenizing
                tokens = tokenizer.tokenize(line)
                #Filtering out stopwords along with stemming
                stemmed = [stemmer.stem(str(t)) for t in tokens if not t in stop_words]
                #Once words are stemmed, finding conditional probability of each word in the file using laplace smoothing of 1
                for word in stemmed:
                    if word in no_of_unique_ham_words_in_ham_folder:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham_folder[word] + 1)/(total_no_of_ham_words_in_ham_folder +
                             no_of_unique_words_in_whole_train_folder))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_no_of_ham_words_in_ham_folder + no_of_unique_words_in_whole_train_folder))
                    
                    if word in no_of_unique_spam_words_in_spam_folder:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam_folder[word] + 1)/(total_no_of_spam_words_in_spam_folder +
                               no_of_unique_words_in_whole_train_folder))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_no_of_spam_words_in_spam_folder + no_of_unique_words_in_whole_train_folder))
            
            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        #In test/ham folder if probability of ham > spam then the predicted class will be chosen as ham during test time
        if (probability_of_ham > probability_of_spam):
            correct_guesses = correct_guesses + 1

    path_for_spam_test = glob.glob("test/spam/*.txt")
    test_file_count += len(path_for_spam_test)

    #Calculating number of correct guesses in test/spam during test time and determining the accuracy without stopwords
    for file in path_for_spam_test:
        probability_of_ham = 0
        probability_of_spam = 0
        with open(file, 'r') as fp:
            #Reading each line of the file
            for line in fp:
                #tokenizing
                tokens = tokenizer.tokenize(line)
                #Filtering out stopwords along with stemming
                stemmed = [stemmer.stem(str(t)) for t in tokens if not t in stop_words]
                #finding conditional probability of each word in the file using laplace smoothing of 1
                for word in stemmed:
                    if word in no_of_unique_ham_words_in_ham_folder:
                        probability_of_ham = probability_of_ham + math.log10((no_of_unique_ham_words_in_ham_folder[word] + 1) / (total_no_of_ham_words_in_ham_folder +
                             no_of_unique_words_in_whole_train_folder))
                    else:
                        probability_of_ham = probability_of_ham + math.log10((1) / (total_no_of_ham_words_in_ham_folder + no_of_unique_words_in_whole_train_folder))
                    
                    if word in no_of_unique_spam_words_in_spam_folder:
                        probability_of_spam = probability_of_spam + math.log10((no_of_unique_spam_words_in_spam_folder[word] + 1) / (total_no_of_spam_words_in_spam_folder +
                               no_of_unique_words_in_whole_train_folder))
                    else:
                        probability_of_spam = probability_of_spam + math.log10((1) / (total_no_of_spam_words_in_spam_folder + no_of_unique_words_in_whole_train_folder))
            
            probability_of_ham = probability_of_ham_class + probability_of_ham
            probability_of_spam = probability_of_spam_class + probability_of_spam

        #In test/spam folder if probability of ham < spam then the predicted class will be chosen as spam during test time
        if probability_of_ham < probability_of_spam:
            correct_guesses = correct_guesses + 1

    test_accuracy_without_stopwords = 0
    print "\n Number of correct guesses during test without stop words:\t%d/%s" % (correct_guesses, test_file_count)
    test_accuracy_without_stopwords = (correct_guesses / test_file_count) * 100

    return test_accuracy_without_stopwords

# NaiveBayesWithoutStopWords()
