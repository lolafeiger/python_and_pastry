import csv
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

#read in data, convert time strings to time times, and select only english entries with decoded text
twitterdata = pd.read_csv('/Users/lolafeiger/Documents/cronuts/cronut 20th.csv')
from dateutil.parser import parse
from datetime import datetime
time_test = [datetime.strptime(x, '%d/%m/%Y %H:%M:%S').date() for x in twitterdata.time]
twitterdata['date'] = pd.Series(time_test)

# just selecting the english language tweets, and decoding the text
twitterdata = twitterdata[twitterdata['user_lang'] == 'en']
twitterdata.text = twitterdata.text.map(lambda x : x.decode('utf-8', errors='replace'))

#removing "loc: " component from geo_coordinates column to map geolocated tweets
fulltwitterdata = pd.read_csv('/Users/lolafeiger/Documents/fullpass.csv')
fulltwitterdata['geo_coordinates'] = fulltwitterdata['geo_coordinates'].map(lambda x: str(x)[6:])
#exporting file to load in GIS
df = pd.DataFrame(fulltwitterdata)
df.to_csv("fulltwitterdata.csv")

#SENTIMENT ANALYSIS OF TWITTER DATA WITH SUPERVISED CLASSIFICATION
Location = r'/Users/lolafeiger/Documents/training19th.csv'
bigtrain = read_csv(Location)
Location1 = r'/Users/lolafeiger/Documents/test19th.csv'
bigtest = read_csv(Location1)
#bigtrain1 = pd.read_csv('/Users/lolafeiger/Documents/training19th.csv')
#bigtest1 = pd.read_csv('/Users/lolafeiger/Documents/test19th.csv')

bigtrain1.text = bigtrain1.text.map(lambda x : x.decode('utf-8', errors='replace'))
bigtest1.text = bigtest1.text.map(lambda x : x.decode('utf-8', errors='replace'))
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(bigtrain1.text)
X_test = vectorizer.transform(bigtest1.text)
cross_val_score(MultinomialNB(), X_train, bigtrain1.sentiment, score_func=auc_score)
#Out[45]: array([ 0.75724638,  0.73434343,  ])
cross_val_score(LogisticRegression(), X_train, bigtrain1.sentiment, score_func=auc_score)
#Out[46]: array([ 0.74919485,  0.72626263,  0.76296296])
cross_val_score(RandomForestClassifier(), X_train.toarray(), bigtrain1.sentiment, score_func=auc_score) 
#array([ 0.67673108,  0.76868687,  0.81666667])
from sklearn.svm import SVC 
cross_val_score(SVC(), X_train.toarray(), bigtrain1.sentiment, score_func=auc_score)
#Out[115]: array([ 0.5,  0.5,  0.5])
model = MultinomialNB().fit(X_train, list(bigtrain1.sentiment))
predictions = model.predict_proba(X_test)[:,1]

#if i had wanted to print the predicted values for my test set on a separate csv
#submission = pd.DataFrame({'user id':bigtest.from_user, 'sentiment': predictions})
#submission.to_csv('predictions.csv', index=False)

#if i had wanted to map the predictions to the actual labeled sentiment:
#print predictions
#print zip(bigtest1.sentiment, predictions)

#predictions = model.predict(X_test)
#auc score for predictions against actual test labels
#print auc_score(bigtest1.sentiment, predictions)
#0.59868959869

predictions.mean()
#Out[41]: 0.56074766355140182

#SAME MODEL, MORE LABELS
training2  = pd.read_csv('/Users/lolafeiger/Documents/training2.csv')
test2  = pd.read_csv('/Users/lolafeiger/Documents/test2.csv')
training2.text = training2.text.map(lambda x : x.decode('utf-8', errors='replace'))
test2.text = test2.text.map(lambda x : x.decode('utf-8', errors='replace'))

vectorizer1 = CountVectorizer()
X_train = vectorizer1.fit_transform(training2.text)
X_test = vectorizer1.transform(test2.text)

cross_val_score(MultinomialNB(), X_train, training2.sentiment, score_func=auc_score)
#array([ 0.70361446,  0.72111977,  0.72631877])
cross_val_score(LogisticRegression(), X_train, training2.sentiment, score_func=auc_score)
#array([ 0.73288448,  0.7265769 ,  0.72518434])
cross_val_score(RandomForestClassifier(), X_train.toarray(), training2.sentiment, score_func=auc_score) 
#array([ 0.67980156,  0.63203402,  0.70944413])
cross_val_score(SVC(), X_train.toarray(), training2.sentiment, score_func=auc_score)
#array([ 0.5,  0.5,  0.5])

model = MultinomialNB().fit(X_train, list(training2.sentiment))
predictions = model.predict_proba(X_test)[:,1]

predictions.mean()
#0.43601895734597157

#this prints a full mapping of predicted scores to actual labels in the test set
#unable to complete this part - issues with relabeled files. not necessary to pursue,
#using the first model with better AUC scores
zip(test2.sentiment, predictions)

#USING MODEL 1 TO MAP TO DATA ARCHIVE MEAN SENTIMENT ESTIMATE BY DAY
fulltwitterdata.text = fulltwitterdata.text.map(lambda x : x.decode('utf-8', errors='replace'))
oos_data = vectorizer.transform(fulltwitterdata.text)
predictions = model.predict_proba(oos_data)[:,1]
predictions 
#array([ 0.00343251,  0.99931768,  0.99999935, ...,  0.00503093,
#        0.2680684 ,  0.62987254])

fulltwitterdata['predicted_sentiment'] = pd.Series(predictions)
grouped = fulltwitterdata.groupby('date')

In [80]: for group, val in grouped:
   ....:     print group, val['predicted_sentiment'].mean()
   ....:     
#2013-07-19 0.43685281327
#2013-07-20 0.456406444985
#2013-07-21 0.468320084743
#2013-07-22 0.438766235644
#2013-07-23 0.488921847032
#2013-07-24 0.501509426037
#2013-07-28 0.423878039783
#2013-07-29 0.495397181119
#2013-07-31 0.505528431549

#writing this sentiment estimate output to csv file to load into R
f = open('predicted_means.csv', 'w')
writer1 = csv.writer(f)

In [86]: for group, val in grouped:
    writer1.writerow([group, val['predicted_sentiment'].mean()])
   ....:     
f.close()

#TWITTER ANALYSIS FOR VISUALIZATIONS
from dateutil.parser import parse
from datetime import datetime

#setup 
fulltwitterdata = pd.read_csv('/Users/lolafeiger/Documents/fullpass2.csv')
fulltwitterdata = fulltwitterdata[fulltwitterdata['user_lang'] == 'en']
[datetime.strptime(x, '%d/%m/%Y %H:%M:%S').date() for x in fulltwitterdata.time]
time_test = [datetime.strptime(x, '%d/%m/%Y %H:%M:%S').date() for x in fulltwitterdata.time]
print time_test[:10]
#[datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20), datetime.date(2013, 7, 20)]
fulltwitterdata['date'] = pd.Series(time_test)

#tweet counts by day
grouped = fulltwitterdata.groupby('date')

In [65]: for group, val in grouped:
   ....:     print group, val['text'].count()
   ....:     
#2013-07-19 1148
#2013-07-20 1601
#2013-07-21 831
#2013-07-22 879
#2013-07-23 798
#2013-07-24 76
#2013-07-28 455
#2013-07-29 841
#2013-07-31 664

#writing this count output to csv to import to R
f = open('data_tweet_counts.csv', 'w')
writer = csv.writer(f)
In [68]: for group, val in grouped:
   ....:     writer.writerow([group, val['text'].count()])
   ....:     
f.close()

#generating list of top terms per day 
def get_top_terms(text):
    top_term_vectorizer = CountVectorizer(stop_words='english')
    top_term_data = top_term_vectorizer.fit_transform(text)
    top_terms = sorted(zip(top_term_data.toarray().sum(axis=0), top_term_vectorizer.get_feature_names()), reverse=True)[:10]
    return top_terms
   ....: 

#writing this top term output to csv to import to R
output_file = open('filename.csv', 'w')
fancy_new_writer = csv.writer(output_file)
grouped = fulltwitterdata.groupby('date')
for group, val in grouped:
    fancy_new_writer.writerow( [group, get_top_terms(val.text)])   ....:     
output_file.close()

#more refined method to extract top ten terms, with each term accorded own column
real_new = open('newfile.csv', 'w')
real_new_writer = csv.writer(real_new)

grouped = fulltwitterdata.groupby('date')

for group, val in grouped:
   real_new_writer.writerow([group] + list(get_top_terms(val.text)))
   .....:         
real_new.close()

#top hashtags
tweet1  = fulltwitterdata.text[1]
hashtags = fulltwitterdata.text.map( lambda tweet1 : [x for x in tweet1.split() if x.startswith('#')])
hashtag_tag_vectorizer = CountVectorizer()
hashtags = fulltwitterdata.text.map( lambda tweet1 : " ".join([x for x in tweet1.split() if x.startswith('#')]))
hash_tag_data = hashtag_tag_vectorizer.fit_transform(hashtags)
#lists all hashtags alphabetically
hashtag_tag_vectorizer.get_feature_names()
#inspects how many
len(hashtag_tag_vectorizer.get_feature_names())
#476

#sorts top ten most used hashtags by frequency, descending
sorted(zip(hash_tag_data.toarray().sum(axis=0), hashtag_tag_vectorizer.get_feature_names()), reverse=True)[:10]
#Out[119]: 
#(1976, u'cronut'),
#(88, u'cronuts'),
#(77, u'rainroom'),
#(75, u'nyc'),
#(66, u'food'),
#(61, u'likecrack'),
#(55, u'moma'),
#(48, u'foodie'),
#(44, u'frozensmore'),
#(37, u'yum')]

#writing this output to a csv to load into R
hashtag_tags = sorted(zip(hash_tag_data.toarray().sum(axis=0), hashtag_tag_vectorizer.get_feature_names()), reverse=True)[:10]

hashtag_file = open('hashtags.csv', 'w')
hashtag_writer = csv.writer(hashtag_file)
hashtag_writer.writerow(hashtag_tags)
hashtag_file.close()
