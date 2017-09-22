library(readr)
library(class)
library(randomForest)

train <- read_csv(".../Digit_Recognizer_Train.csv")
test <- read_csv(".../Digit_Recognizer_Test.csv")

nrow(train)
ncol(train)
labels <- factor(train[nrow(train),1])
numTrees = 15

rFor <- randomForest(train[,-1],train$label,xtest=test,ntree=numTrees)

submission <- data.frame(ImageId=1:nrow(test),Label=round(rFor$test$predicted))

head(submission)

write.csv(submission, ".../randomForest_1.csv")
