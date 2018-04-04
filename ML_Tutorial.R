###### FriData Machine Learning Tutorial

# Load Packages
library(kknn)
library(tree)
library(randomForest)
library(rpart)

# Load data
caldata <- read.csv("https://raw.githubusercontent.com/jonathan-chua/Kenway_FriData/Intro_to_Machine_Learning/CaliforniaHousing.csv")

# Look at fields
head(caldata)

# Hypothesize an initial relationship
# Initial thought is that there is a relationship between Median Income and log Median Value
# Understanding that this isn't the full method to do regression analysis, just a benchmark
plot(caldata$medianIncome, caldata$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgray",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

# To properly test accuracy, we need to set up train, test, and validation sets
# Train sets are used to develop the models
# Test sets are used for model selection and tuning
# Validation sets are the final proving ground for a model
# We'll go with 60% train, 20% test, 20% validation
set.seed(1)
# Number of records
n <- nrow(caldata)
# Randomly select 60% of the observations
train.obs <- sample.int(n, n*0.60)
# Make a dataframe with the training set
cal.train <- caldata[train.obs, ]
# Get remaining data for test and validate
cal.test_val <- caldata[-train.obs, ]
n.test_val <- nrow(cal.test_val)
test.obs <- sample.int(n.test_val, n.test_val*0.50)
cal.test <- cal.test_val[test.obs, ]
cal.val <- cal.test_val[-test.obs, ]

# Run Linear Regression
lr.cal <- lm(logMedVal ~ medianIncome, data=cal.train)

abline(lr.cal, col="red", lwd=2)

# Ok fit, there are issues with the data (top-coded / upper bounded) but it fits the general trend

# We'll measure accuracy using Mean Squared Error (MSE)
# MSE = mean((prediction_i - actual_i)^2)

# Create a function because we'll be using this a lot
calc.mse <- function(yhat, y) {
  e <- yhat - y
  se <- e^2
  return(mean(se))
}

# Create a table to store performance
model.perf <- matrix(NA, ncol=0, nrow=0)

# Get predictions for the data (skipping to validation set because no tuning)
lr.yhat <- predict(lr.cal, newdata=cal.val)

# Calculate MSE
lr.mse <- calc.mse(lr.yhat, cal.val$logMedVal)

print(paste("Linear Regression MSE = ", round(lr.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="Linear Regression", mse=lr.mse))

# k-NN Regression
# Plot initial effort
# Create sample Median Incomes
sample.income <- data.frame(medianIncome=sort(caldata$medianIncome))

# Run for Median Income only w/ k=10
knn.cal <- kknn(logMedVal ~ medianIncome, train=cal.train, test=sample.income, k=10, kernel="rectangular")

lines(sample.income$medianIncome, knn.cal$fitted.values, col="blue", lwd=2)

# Get Accuracy for Median Income and full models
knn.calInc <- kknn(logMedVal ~ medianIncome, train=cal.train, test=cal.val, k=10, kernel="rectangular")

knnInc.yhat <- knn.calInc$fitted.values

knnInc.mse <- calc.mse(knnInc.yhat, cal.val$logMedVal)

print(paste("10-NN w/ Median Income = ", round(knnInc.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="10-NN, Median Income", mse=knnInc.mse))
# Actually worse than Linear Regression

knn.calFull <- kknn(logMedVal ~ . , train=cal.train, test=cal.val, k=10, kernel="rectangular")

knnFull.yhat <- knn.calFull$fitted.values

knnFull.mse <- calc.mse(knnFull.yhat, cal.val$logMedVal)

print(paste("10-NN w/ All Fields = ", round(knnFull.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="10-NN, All Fields", mse=knnFull.mse))
# Significantly better performance

# Tuning - picking the best k to optimize prediction
# This is where train & test are used, validation is left out so that it is completely independent of our tuning

# Capcture nn performance
nn.mse <- matrix(NA, ncol=0, nrow=0)

# Run model for k from 1 to 100, capturing the MSE for each value
for (nn in 1:100) {
  knn.train <- kknn(logMedVal ~ . , train=cal.train, test=cal.test, k=nn, kernel="rectangular")
  mse.train <- calc.mse(knn.train$fitted.values, cal.test$logMedVal)
  nn.mse <- rbind(nn.mse, data.frame(k=nn, mse=mse.train))
}

# Plot the MSE to find the best performing one
plot(nn.mse$k, nn.mse$mse, type="l", col="navy", lwd=2,
     main="MSE by k",
     xlab="k",
     ylab="MSE")

min.k <- nn.mse[which.min(nn.mse$mse), ]

points(min.k$k, min.k$mse, col="red", pch=16)

print(paste("Optimal Value k = ", min.k$k, sep=""))

# Run optimal model & get estimates...
# Chart plot the fit against the data, remember to sort the test set by Median Income so it matches the chart
# If possible, try and find the point with the smallest k that is statistically indistinguishable from the minimum point
# That is, the MSE - 1 standard deviation of the error (or squared error)  <= the MSE of the minimum point
