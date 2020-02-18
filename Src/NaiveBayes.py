from NaiveBayesWithStopWords import NaiveBayesWithStopWords
from NaiveBayesWithoutStopWords import NaiveBayesWithoutStopWords

#Calaculates accuracy of NaiveBayes with stop words
training_accuracy,test_accuracy_with_stopwords = NaiveBayesWithStopWords()

#Calculates accuracy of  NaiveBayes after filtering out stop words
test_accuracy_without_stopwords = NaiveBayesWithoutStopWords()

print "\n Training    Test_with_stopwords    Test_without_stopwords"
print ' %.2f       %.2f		    %.2f'%(training_accuracy,test_accuracy_with_stopwords,test_accuracy_without_stopwords)
