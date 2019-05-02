rm(list=ls())
#setting working directory
setwd("D:/Edwisor/Project 1 - Churn Reduction/R")
getwd()

#importing libraries
library('corrgram')
library('caret')
library('C50')
library('rpart')
library('randomForest')
library('caTools')
library('e1071')

#loading the data
train_df = read.csv("Train_data.csv")
test_df = read.csv("Test_data.csv")

#checking summary
summary(train_df)

#checking count of missing values in each column
sapply(train_df, function(x) sum(is.na(x)))
sapply(test_df, function(x) sum(is.na(x)))

#checking the datatype of each column
sapply(train_df, function(x) class(x))
sapply(test_df, function(x) class(x))

#converting area code to factor
train_df$area.code = as.factor(train_df$area.code)

#column names and data of categorical columns
factor_ind = sapply(train_df, is.factor)
factor_train = train_df[, factor_ind]

#chi-square test of independence
for (i in 1:5){
  print(names(factor_train)[i])
  print(chisq.test(table(train_df$Churn, factor_train[,i])))
}

#p-value of phone number and area code is greater than 0.05, hence dropping them
train_df = train_df[-c(3,4)]
test_df = test_df[-c(3,4)]

#column names of numerical columns
numeric_ind = sapply(train_df, is.numeric)

#correlation plot
corrgram(train_df[,numeric_ind], order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

#removing numeric variables with high correlation
train_df = train_df[-c(6,9,12,15)]
test_df = test_df[-c(6,9,12,15)]


#label encoding categorical variables
for(i in 2:ncol(train_df)){
  if(class(train_df[,i]) == 'factor'){
    train_df[,i] = factor(train_df[,i], labels = (1:length(levels(factor(train_df[,i])))))
  }
}

for(i in 2:ncol(test_df)){
  if(class(test_df[,i]) == 'factor'){
    test_df[,i] = factor(test_df[,i], labels = (1:length(levels(factor(test_df[,i])))))
  }
}

#dummy encoding
dmy = dummyVars("~ state", data = train_df)
train_df_state = data.frame(predict(dmy,newdata = train_df))

dmy = dummyVars("~ state", data = test_df)
test_df_state = data.frame(predict(dmy,newdata = test_df))

train_df = cbind(train_df,train_df_state)
test_df = cbind(test_df,test_df_state)

train_df = train_df[-c(1)]
test_df = test_df[-c(1)]


y_train = train_df$Churn
y_test = test_df$Churn
x_train = train_df[-c(14)]
x_test = test_df[-c(14)]

##DECISION TREE MODEL
DT_model = rpart(y_train ~., data = x_train, method = "class")
DT_pred = predict(DT_model, type = "class", newdata = x_test)

DT_confusion_matrix = table(y_test, DT_pred)

TN = DT_confusion_matrix[1,1]
TP = DT_confusion_matrix[2,2]
FN = DT_confusion_matrix[2,1]
FP = DT_confusion_matrix[1,2]

#accuracy of Decision tree = 94.36%
accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)
#false negative rate of Decision tree = 35.26%
FNR = (FN*100)/(FN+TP)
#true negative rate or specificity of Decision tree = 98.96%
specificity = (TN*100)/(TN+FP)
#true positive rate or recall or sensitivity of Decision tree = 64.73% 
recall = (TP*100)/(TP+FN)

##RANDOM FOREST MODEL
RF_model = randomForest(y_train ~., x_train, importance = TRUE, ntree = 300)
RF_pred = predict(RF_model, x_test)

RF_confusion_matrix = table(y_test, RF_pred)

TN = RF_confusion_matrix[1,1]
TP = RF_confusion_matrix[2,2]
FN = RF_confusion_matrix[2,1]
FP = RF_confusion_matrix[1,2]

#accuracy of Random Forest = 94.96%
accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)
#false negative rate of Random Forest = 37.05%
FNR = (FN*100)/(FN+TP)
#true negative rate or specificity of Random Forest = 99.86%
specificity = (TN*100)/(TN+FP)
#true positive rate or recall or sensitivity of Random Forest = 62.94% 
recall = (TP*100)/(TP+FN)

##LOGISTIC REGRESSION MODEL
logit_model = glm(y_train ~., data = x_train, family = "binomial")

summary(logit_model)

logit_pred = predict(logit_model, newdata = x_test, type = "response")
#convert probabality
logit_pred = ifelse(logit_pred > 0.5, 2, 1)

logit_confusion_matrix = table(y_test, logit_pred)

TN = logit_confusion_matrix[1,1]
TP = logit_confusion_matrix[2,2]
FN = logit_confusion_matrix[2,1]
FP = logit_confusion_matrix[1,2]

#accuracy of Logistic Regression = 87.16%
accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)
#false negative rate of Logistic Regression = 75%
FNR = (FN*100)/(FN+TP)
#true negative rate or specificity of Logistic Regression = 96.81%
specificity = (TN*100)/(TN+FP)
#true positive rate or recall or sensitivity of Logistic Regression = 25% 
recall = (TP*100)/(TP+FN)

##NAIVE BAYES MODEL
NB_model = naiveBayes(y_train ~., data = x_train)

NB_pred = predict(NB_model, x_test, type = "class")

NB_confusion_matrix = table(y_test, NB_pred)

TN = NB_confusion_matrix[1,1]
TP = NB_confusion_matrix[2,2]
FN = NB_confusion_matrix[2,1]
FP = NB_confusion_matrix[1,2]

#accuracy of Naive Bayes = 56.68%
accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)
#false negative rate of Naive Bayes = 45.98%
FNR = (FN*100)/(FN+TP)
#true negative rate or specificity of Naive Bayes = 57.10%
specificity = (TN*100)/(TN+FP)
#true positive rate or recall or sensitivity of Naive Bayes = 54.01% 
recall = (TP*100)/(TP+FN)
