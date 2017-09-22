library(readr)
library(class)

train <- read_csv("/Users/nwalker018/Desktop/Datasets/Digit Recognizer/Digit_Recognizer_Train.csv")

test <- read_csv("/Users/nwalker018/Desktop/Datasets/Digit Recognizer/Digit_Recognizer_Test.csv")


train <- train[1:3000,]

dim(train)
dim(test)

pc <- proc.time()
Knn <- knn(train[,-1], test,train$label,k=10)
proc.time() - pc

submission <- data.frame(ImageId=1:nrow(test),Label=Knn)
head(submission)
dim(submission)

write_csv(submission,"/Users/nwalker018/Desktop/Datasets/Digit Recognizer/KNN_Submission_1.csv")
