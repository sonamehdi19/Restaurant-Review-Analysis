library(data.table)
library(tidyverse)
library(inspectdf)
library(text2vec)
library(caTools)
library(glmnet)

#Importing data
df<-fread("nlpdata.csv")

#Data understanding----
df %>% dim()

df %>% colnames()

df %>% glimpse()

df %>% inspect_na()

df$V1<-df$V1 %>% as.character()    #id column converted to character, as it contains no importance as numeric 

# Splitting data into train and test set----
set.seed(123)
split <- df$Liked %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

#Tokenizing----
train %>% colnames()

it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$V1,
         progressbar = F) 

#Creating vocabulary---- 
vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>%   
  head(110) %>% 
  tail(10) 

#Vectorizing 
vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)


dtm_train %>% dim()

identical(rownames(dtm_train), train$V1)

# Modeling Normal Nfold GLm ----
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#AUC score for train set 
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$V1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)

# Pruning some words with defining stopwords while creating vocabulary for removal----
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

#Creating  DTM for Training and Testing with new pruned vocabulary----
vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#AUC for training
glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#AUC score for test 
glmnet:::auc(test$Liked, preds) %>% round(2)
#As there is not a significant difference between AUC scores for training and test sets, it indicates that there is no overfitting


