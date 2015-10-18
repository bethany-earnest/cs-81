#INCLUDED BELOW: CLASSIFICATION USING KMEANS & CLASSIFICATION USING SUPPORT VECTOR MACHINES


##CLASSIFICATION USING KMEANS

from numpy import array
from numpy import asarray
from pandas import DataFrame
from pandas.io import gpq
import scipy.cluster.vq import vq, kmeans, whiten

####Define query: this is stored elsewhere in the book, let me know if you need me to go grab it. 
query=""" 
SELECT word, 
SUM ( if (corpus== '1kinghenryiv', tfidf, 0)) as onekinghenryiv, 
...
SUM (if (corpus=='winterstale', tfidf, 0)) as winterstale, 
FROM [ch13.shakespear_tfidf]
GROUP BY word
"""

data_frame= gbq.read_gbq(query)

####drop words from dataframe
del data_frame['word']

####convert features to array to run kmeans clustering
features=asarray(data_frame.T)

####pass array to kmeans
codes,_ = kmeans(features,2)

####assign shakespear plays to clusters
assignments,_=vg(features, codes)

####print results
results={'play' : array(data_frame.columns.values), 'cluster' : assignments}

results_frame=DataFrame.from_dict(results).sort(['cluster', 'play'])




##CLASSIFICATION USING SUPPORT VECTOR MACHINE

####define function words
function_words = ["a", "able", "aboard", "about", "above", "absent", "according" , "accordingly", 
                  "across", "after", "against", "ahead", "albeit", "all", "along", "alongside", "although", 
                  "am", "amid", "amidst", "among", "amongst", "amount", "an", "and", "another", "anti", "any", 
                  "anybody", "anyone", "anything", "are", "around", "as", "aside", "astraddle", "astride", "at", 
                  "away", "bar", "barring", "be", "because", "been", "before", "behind", "being", "below", "beneath", 
                  "beside", "besides", "better", "between", "beyond", "bit", "both", "but", "by", "can", "certain", 
                  "circa", "close", "concerning", "consequently", "considering", "could", "couple", "dare", "deal", 
                  "despite", "down", "due", "during", "each", "eight", "eighth", "either", "enough", "every", "everybody", 
                  "everyone", "everything", "except", "excepting", "excluding", "failing", "few", "fewer", "fifth", 
                  "first", "five", "following", "for", "four", "fourth", "from", "front", "given", "good", "great", 
                  "had", "half", "have", "he", "heaps", "hence", "her", "hers", "herself", "him", "himself", "his", 
                  "however", "i", "if", "in", "including", "inside", "instead", "into", "is", "it", "its", "itself", 
                  "keeping", "lack", "less", "like", "little", "loads", "lots", "majority", "many", "masses", "may", 
                  "me", "might", "mine", "minority", "minus", "more", "most", "much", "must", "my", "myself", "near", 
                  "need", "neither", "nevertheless", "next", "nine", "ninth", "no", "nobody", "none", "nor", "nothing", 
                  "notwithstanding", "number", "numbers", "of", "off", "on", "once", "one", "onto", "opposite", "or", 
                  "other", "ought", "our", "ours", "ourselves", "out", "outside", "over", "part", "past", "pending", 
                  "per", "pertaining", "place", "plenty", "plethora", "plus", "quantities", "quantity", "quarter", 
                  "regarding", "remainder", "respecting", "rest", "round", "save", "saving", "second", "seven", "seventh", 
                  "several", "shall", "she", "should", "similar", "since", "six", "sixth", "so", "some", "somebody", 
                  "someone", "something", "spite", "such", "ten", "tenth", "than", "thanks", "that", "the", "their", 
                  "theirs", "them", "themselves", "then", "thence", "therefore", "these", "they", "third", "this", "those", 
                  "though", "three", "through", "throughout", "thru", "thus", "till", "time", "to", "tons", "top", "toward", 
                  "towards", "two", "under", "underneath", "unless", "unlike", "until", "unto", "up", "upon", "us", "used", 
                  "various", "versus", "via", "view", "wanting", "was", "we", "were", "what", "whatever", "when", "whenever", 
                  "where", "whereas", "wherever", "whether", "which", "whichever", "while", "whilst", "who", "whoever", 
                  "whole", "whom", "whomever", "whose", "will", "with", "within", "without", "would", "yet", "you", "your", 
                  "yours", "yourself", "yourselves"]

####function word counter
from sklearn.feature_extraction.text import CountVectorizer
extractor=CountVectorizer(vocabulary=function_words)

####classification using support vector machine
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import grid_search

parameters={'kernel':('linear','rbf'),'C':[1,10]}
svr=SVC()
grid=grid_search.GridSearchCV(svr, parameters)

pipeline1=Pipeline([('feature_extraction',extractor), ('clf', grid)])

####extracting character n-grams
pipeline=Pipeline([('feature_extraction', CountVectorizer(analyzer='char', ngram_range=(3,3))),('classifier',grid)])

####score results as a percentage of correct assignments
scores=cross_val_score(pipeline, documents, classes, scoring='f1')
