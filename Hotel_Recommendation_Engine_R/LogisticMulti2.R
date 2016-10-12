library(R.matlab)
data<-read.csv("testing-5.csv")
myData=as.matrix(data)
X = myData[,c(1:35)]
m=nrow(X)
onesMatrix = matrix(1,m,1)
X = cbind(onesMatrix,X)
y = myData[,c(36)]

#%% ============ Part 2: Vectorize Logistic Regression ============
lambda = 0.1;
num_labels = 29;
source("prediction2.R")
all_theta = oneVsAll(X, y, num_labels, lambda)
pred = predictOneVsAll(all_theta, X)
training_accurac=mean((pred[,2] == y)) * 100
