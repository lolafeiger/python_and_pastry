#install.packages('lubridate')
#install.packages('wordcloud')
#install.packages('tm')
#install.packages('RColorBrewer')
#install.packages('RWeka')
#install.packages('lattice')
#install.packages('ggplot2')
#install.packages('stringr')

#importing csv files crafted in python: one for predicted mean sentiment score by day and the other
#for tweet counts by day
predicted_means <- read.csv('/Users/lolafeiger/Documents/predicted_means.csv', header=F)
tweet_counts <- read.csv('/Users/lolafeiger/Documents/data_tweet_counts.csv', header = F) 

#plotting tweet_counts
ggplot(data=tweet_counts) + aes(x=Date, y=TweetsCount) + geom_bar() + xlab("Date") + ylab("Number of Tweets") + ggtitle("Summer of the Cronut: Tweet Volume per Day") + theme_bw()  

#adding columns to use for fill color and labels for sentiment score bars, in acccordance with ggplot2 format
names(predicted_means) <- c("Date", "PredictedMean")  
cynicism_meter <- (0,0,1,0,1,1,0,1,1) 
predicted <- cbind(predicted_means, cynicism_meter)
cynicism <- c("not cynical", "not cynical", "cynical", "not cynical", "cynical", "cynical", "not cynical", "cynical", "cynical")
predicted <- cbind(predicted, cynicism)
cynicism1 <- c("", "", "cynical", "", "cynical", "cynical", "", "cynical", "cynical") 
predicted <- cbind(predicted, cynicism1)
p <- ggplot(data=predicted_means) + aes(x=Date, y=PredictedMean, fill=factor(cynicism_meter)) + scale_fill_manual(values=c("#FFCC66", "#669933")) + geom_bar() + geom_text(aes(label=cynicism), vjust=1.7, hjust=2, angle=90, size=6, colour="white") + xlab("Date") + ylab("Sentiment Estimate") + ggtitle("Summer of the Cronut: Tweet Sentiment by Day") + theme_bw()
#removing the legend
p + guides(fill=F)

#word cloud!
#text processing through to the level needed for most frequent terms and wordclouds
# get the text
some_txt = twitterdata$text #sapply(some_tweets, function(x) x$text)
# remove retweet entities
some_txt = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", some_txt)
# remove at people
some_txt = gsub("@\\w+", "", some_txt)
# remove punctuation
some_txt = gsub("[[:punct:]]", "", some_txt)
# remove numbers
some_txt = gsub("[[:digit:]]", "", some_txt)
# remove html links
some_txt = gsub("http\\w+", "", some_txt)
# remove unnecessary spaces
some_txt = gsub("[ \t]{2,}", "", some_txt)
some_txt = gsub("^\\s+|\\s+$", "", some_txt)
some_txt = tolower(some_txt)

words <- unlist(strsplit(some_txt, " "))
#words <- words[!(words %in% stopwords("en"))] # remove stop words

#customizing 'tm' package's list of english stopwords (which doesn't remove 'the'!!)
myStopwords <- c(stopwords("en"), 'I', 'The', 'the', 'i', 'a', 'A', 'im', 'Im', 'Up', 'up', 'de', 'its', 'what', 'What', 'if', 'If', 'can', 'just', 'get') 
words <- words[!(words %in% myStopwords)]

#plotting a word cloud
wordstable <- as.data.frame(table(words))
wordstable <- wordstable[order(wordstable$Freq, decreasing=T), ]
wordstable <- wordstable[-1, ]
head(wordstable)
#words Freq
#1090       craze 5958
#3532         new 5238
#1144   croissant 4878
#602        black 3096
#2866        just 2880
#188  advertisers 2826

#png(paste(searchTerm, "wordcloud.png", sep="--"), w=800, h=800)
wordcloud(wordstable$words, wordstable$Freq, scale = c(8, .2), min.freq = 3, max.words = 200, random.order = FALSE, rot.per = .15, colors = brewer.pal(8, "Dark2"))
dev.off()


--------------------------------------
#R scripts not used in final run-through, but experimented with during course of the project  
#my attempt at top terms by day, prior to finding solution in python
twitterdata <- read.csv('/Users/lolafeiger/Documents/cronut july 23.csv')
View(twitterdata)
cronut.tweets <- twitterdata$text
length(cronut.tweets)
#[1] 3230
class(cronut.tweets)
#[1] "factor"
table(twitterdata$user_lang)
#   ca    de    en    es    fr    id    it    ja    ko    nl    pt    ru    sv    th xx-lc zh-cn 
#9     8  3008   107    46     1     6     3     5    30     1     1     1     1     2     1 
#bucketing tweets by date
July_20 <- subset(newdata,twitterdata2$time=="2013-07-20") 
#View(July_20)
length(July_20$text)
#[1] 817
July_21 <- subset(newdata,twitterdata2$time=="2013-07-21") 
length(July_21$text)
#[1] 456
July_22 <- subset(newdata,twitterdata2$time=="2013-07-22") 
length(July_22$text)
#[1] 808
July_23 <- subset(newdata,twitterdata2$time=="2013-07-23") 
length(July_23$text)
#[1] 847
July_24 <- subset(newdata,twitterdata2$time=="2013-07-24") 
length(July_24$text)
#[1] 80
July20 <- length(July_20$text)
#July20
#[1] 817
July21 <- length(July_21$text)
July22 <- length(July_22$text)
July23 <- length(July_23$text)
July24 <- length(July_24$text)
#tweets_per_day <- cbind(July20, July21, July22, July23, July24)
#View(tweets_per_day)  
#mean(tweets_per_day)
#[1] 601.6
tweets_per_day1 <- rbind(July20, July21, July22, July23, July24)
#View(tweets_per_day1)
#visualizing tweets per day
plot(tweets_per_day1)
tweets_per_day2 <- as.data.frame(tweets_per_day1)
qplot(, V1, data=tweets_per_day2)

labels <- rbind("July20","July21","July22","July23","July24")
as.data.frame(labels)
cbind(labels, tweets_per_day2)
#labels  V1
#July20 July20 817
#July21 July21 456
#July22 July22 808
#July23 July23 847
#July24 July24  80
ggplot(data=tweets_per_day2) + aes(x=labels, y=V1) + geom_histogram() + xlab("Date") + ylab("Number of Tweets") + ggtitle("Summer of the Cronut: Tweet Volume per Day") + theme_bw()

---------------------
#simple sentiment analysis using positive and negative word lists
hu.liu.pos = scan('/Users/lolafeiger/Documents/positive-words.txt', what='character', comment.char=';')
hu.liu.neg = scan('/Users/lolafeiger/Documents/negative-words.txt', what='character', comment.char=';')

#adding cronut-context words to positive and negative word lists
pos.words = c(hu.liu.pos, 'upgrade', 'bomb', 'spooner', 'proud', 'dreams')
neg.words = c(hu.liu.neg, 'wtf', 'wait', 'waiting', 'epicfail', 'hype', 'burnt', 'cracked','whining','glorified', 'mess', 'old', 'bore', 'fucking', 'foodie')

some_txt2 = twitterdata2$text

#sentiment analysis with loaded files
score.sentiment = function(some_txt2, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  scores = laply(some_txt2, function(some_txt2, pos.words, neg.words) {
    
    # clean up sentences with R's regex-driven global substitute, gsub():
    some_txt2 = gsub('[[:punct:]]', '', some_txt2)
    some_txt2 = gsub('[[:cntrl:]]', '', some_txt2)
    some_txt2 = gsub('\\d+', '', some_txt2)
    # and convert to lower case:
    some_txt2 = tolower(some_txt2)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(some_txt2, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    #TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=some_txt2)
  return(scores.df)
}

#testing the sentiment function
sample = c("Not sure which event this week was a bigger deal- meeting Anna Wintour or trying a cronut. @davidbrunonyc", "RT Really? @Forbes: The Cronut craze attracts advertisers and a black market for a glorified @Hostess_Snacks #Twinkies","The Cronut is the most in demand pastry in New York. People wait hours or pay up to $40 on the street for one. http://t.co/QUYMb0m4kg"," If I eat one more cronut I'm gonna die", " I JUST WANT A CRONUT")
result = score.sentiment(sample, pos.words, neg.words)
result
#score
#1     0
#2    -2
#3    -1
#4    -1
#5     0

#colnames(result)
#[1] "score" "text" 
#rownames(result)
#[1] "1" "2" "3" "4" "5"
#result$score
#[1]  0 -2 -1 -1  0

#now applying the sentiment function to all tweets
cronut.scores = score.sentiment(some_txt2, pos.words,neg.words, .progress='text')
#|========================================================================================| 100%
#plot(cronut.scores)

---------------------------------
#new.data.frame <- subset(my.data.frame, my.column == 'something')
#selecting only english language tweets
twitterdata1 <- subset(twitterdata, twitterdata$user_lang=='en')

#converting character dates to R-recognized dates
twitterdata1$time <- as.Date(twitterdata1$time, "%d/%m/%Y %H:%M:%S")
is.Date(twitterdata1$time)
#[1] TRUE
head(twitterdata$time)
#summary(twitterdata1$time)
#Min.      1st Qu.       Median         Mean      3rd Qu.         Max. 
#"2013-07-20" "2013-07-20" "2013-07-22" "2013-07-21" "2013-07-23" "2013-07-24"
#unique(twitterdata2$time)
#[1] "2013-07-20" "2013-07-21" "2013-07-22" "2013-07-23" "2013-07-24"
twitterdata2 <- twitterdata1[order(twitterdata1$time),] 

#selecting only relevant columns of the dataset
#editeddf <- c(2,3,5,6,7,14,15)
#newdata <- twitterdata2[editeddf]
#View(newdata)


----------
#a disasterous attempt at parsing top hashtags...
#top hashtags
#tweets information into data frames
text_df <- as.data.frame(twitterdata2$text)
#get the hashtags
text_hashtags = str_extract_all(text_df$text, "#\\w+") 
#put tags in vector
text_hashtags = unlist(text_hashtags)
#calculate hashtag frequencies
text_tags_freq = table(text_hashtags)
#head(text_tags_freq)
#integer(0)  

#str_extract_all("Hello peopllz! My new home is #crazy gr8! #wow", "#\\S+")
#[[1]]
#[1] "#crazy" "#wow"

text_hashtags = str_extract_all(iconv(twitterdata2$text, from='utf-8', to='ascii'), "#\\S+")
#text_hashtags = twitterdata2[grepl("#\\S+", twitterdata2$text),]$text
#Error in substring(string, start, end) : 
  #invalid multibyte string at '<89><db><cf>@vi<73>itphilly: 8 Philly dessert mash-ups to rival the #cronut: http://t.co/GYVX1rdXpM via @EaterPhilly<89>�'



src <- DataframeSource(data.frame(some_txt))
corpus <- Corpus(src)

#attempting some bigrams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer))



#dtm<-DocumentTermMatrix(corpus)
#dtm <- removeSparseTerms(dtm, 0.99)
#dtmm <- data.frame(as.matrix(dtm))

#corpus <- tm_map(corpus, removeWords, stopwords('english'))
#src <- DataframeSource(data.frame(CommentAll))
#corpus <- Corpus(src)



