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

# Y distribution
hist(caldata$logMedVal, main="Log Median Value Distribution")
# Left Skewed

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

summary(lr.cal)
# Very significant

lr.cal$coefficients
# For each unit increase of Median Income, observe a 0.19 increase in Log Median Value

# Residual Analysis
plot(predict(lr.cal), resid(lr.cal), col="darkgray", xlab="Y_hat", ylab="Residuals", main="Predictions vs. Residuals")
abline(h=0, lty=2, lwd=2, col="blue")
# Very clearly top-coded, but pretty evenly distributed outside of that

# Studentized Residual Analysis
hist(rstudent(lr.cal), col='blue', xlab = 'Studentized Residuals', main = 'Studentized Residuals')
# Right skewed but normal, likely due to top coding

## Q-Q Plot
qqnorm(rstudent(lr.cal), col = 'blue', pch = 16, xlab = 'Theoretical Quantiles', ylab = 'Sample Quantiles', main = 'Q-Q Plot')
abline(a=0, b=1)
# Really well fit except for top coding

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

##### k-NN Regression

# Replot
plot(caldata$medianIncome, caldata$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgray",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

# Plot initial effort
# Create sample Median Incomes
sample.income <- data.frame(medianIncome=sort(caldata$medianIncome))

# Run for Median Income only w/ k=10
knn.calInc <- kknn(logMedVal ~ medianIncome, train=cal.train, test=sample.income, k=10, kernel="rectangular")

lines(sample.income$medianIncome, knn.cal$fitted.values, col="blue", lwd=1)

# Get Accuracy for Median Income and full models
knnInc.yhat <- knn.calInc$fitted.values

knnInc.mse <- calc.mse(knnInc.yhat, cal.val$logMedVal)

print(paste("10-NN w/ Median Income = ", round(knnInc.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="10-NN, Median Income", mse=knnInc.mse))
# Actually worse than Linear Regression

# Full data
full.test <- cal.test[order(cal.test$medianIncome), ]

knn.calFull <- kknn(logMedVal ~ . , train=cal.train, test=full.test, k=10, kernel="rectangular")

knnFull.yhat <- knn.calFull$fitted.values

knnFull.mse <- calc.mse(knnFull.yhat, full.test$logMedVal)

print(paste("10-NN w/ All Fields = ", round(knnFull.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="10-NN, All Fields", mse=knnFull.mse))
# Significantly better performance

# Plot full k-NN model
lines(full.test$medianIncome, knn.calFull$fitted.values, col="green", lwd=1)

legend("bottomright", legend=c("Median Income", "Full Model"), fill=c("blue", "green"))  

# Tuning - picking the best k to optimize prediction
# This is where train & test are used, validation is left out so that it is completely independent of our tuning

# SD Function
calc.sde <- function(yhat, y) {
  e <- yhat - y
  se <- e^2
  return(sd(se))
}

# Capture nn performance
nn.mse <- matrix(NA, ncol=0, nrow=0)

# Run model for k from 1 to 100, MSE for each value
for (nn in 1:100) {
  knn.train <- kknn(logMedVal ~ . , train=cal.train, test=cal.test, k=nn, kernel="rectangular")
  mse.train <- calc.mse(knn.train$fitted.values, cal.test$logMedVal)
  nn.mse <- rbind(nn.mse, data.frame(k=nn, mse=mse.train, sde=sde.train))
}

# Plot the MSE to find the best performing one
plot(nn.mse$k, nn.mse$mse, type="l", col="navy", lwd=2,
     main="MSE by k",
     xlab="k",
     ylab="MSE")

min.k <- nn.mse[which.min(nn.mse$mse), ]

points(min.k$k, min.k$mse, col="red", pch=16)

print(paste("Optimal Value k = ", min.k$k, sep=""))

# 5-folds to use SD
# Combine train & test
cal.folds <- rbind(cal.test, cal.train)

# Number of folds
f <- 5
# Total rows
n <- nrow(cal.folds)
# Records per fold
l.fold <- round(n/f,0)

# 5-fold k-NN
fold.nnMSE <- matrix(NA, ncol=0, nrow=0)

for (folds in 1:f) {
  f.start <- if (folds==1) {1} else {f.start+l.fold}
  f.end <- if (folds==1) {f.start+l.fold-1} else {f.end+l.fold}
  i.fold <- f.start:f.end
  f.train <- cal.folds[-i.fold,]
  f.test <- cal.folds[i.fold,]
  
  for (nn in 1:100) {
    f.knn <- kknn(logMedVal ~ . , train=f.train, test=f.test, k=nn, kernel="rectangular")
    f.mse <- calc.mse(f.knn$fitted.values, f.test$logMedVal)
    fold.nnMSE <- rbind(fold.nnMSE, data.frame(k=nn, mse=f.mse, fold=folds))
  }
}

fold.mse <- data.frame(setNames(aggregate(fold.nnMSE$mse, by=list(fold.nnMSE$k), FUN="mean"), c("k","mse")))
fold.sde <- data.frame(setNames(aggregate(fold.nnMSE$mse, by=list(fold.nnMSE$k), FUN="sd"), c("k","sde")))  

plot(fold.mse$k, fold.mse$mse, type="l", col="darkgray", lwd=2,
     main="MSE by k",
     xlab="k",
     ylab="MSE")

points(fold.mse$k, fold.mse$mse, col="darkgray", pch=16)

fmin.k <- fold.mse[which.min(fold.mse$mse), ]

abline(h=fmin.k$mse, lty=2, lwd=1, col="black")

segments(fold.sde$k, fold.mse$mse+fold.sde$sde, fold.sde$k, fold.mse$mse-fold.sde$sde, col="darkgray")

points(fmin.k$k, fmin.k$mse, pch=16, col="red")

f.optimal <- data.frame(fold.mse, mse_sde=fold.mse$mse-fold.sde$sde)

f.optimal2 <- f.optimal[f.optimal$mse_sde <= fmin.k$mse, ]

f.optimal3 <- f.optimal2[which.max(f.optimal2$k),]

points(f.optimal3$k, f.optimal3$mse, pch=18, col="darkgreen", cex=1.5)

print(paste("While k = ", fmin.k$k, " minimizes MSE, it is not statistically distinguishable from k = ", f.optimal3$k,". This will be used for the final calculation", sep=""))

knn.fullFinal <- kknn(logMedVal ~ ., train=cal.train, test=cal.val, k=f.optimal3$k, kernel="rectangular")

knnFinal.mse <- calc.mse(knn.fullFinal$fitted.values, cal.val$logMedVal)

model.perf <- rbind(model.perf, data.frame(model="Optimal k-NN", mse=knnFull.mse))

plot(cal.val$medianIncome, cal.val$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgray",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

tmp.knnFinal <- data.frame(medIncome=cal.val$medianIncome, yhat=knn.fullFinal$fitted.values)
tmp.knnFinal <- tmp.knnFinal[order(tmp.knnFinal$medIncome), ]

lines(tmp.knnFinal$medIncome, tmp.knnFinal$yhat, col="darkgreen")

plot(
    cal.val$logMedVal, 
    knn.fullFinal$fitted.values, 
    xlab="Actual Values", 
    ylab="Predicted Values",
    col="darkgray",
    main="Actual Values vs. Predicted Values"
    )

abline(a=0, b=1, col="red", lty=2, lwd=2)

legend("topleft", legend=c("Perfect Predictions"), fill=c("red")) 

# Run optimal model & get estimates...
# Chart plot the fit against the data, remember to sort the test set by Median Income so it matches the chart
# If possible, try and find the point with the smallest k that is statistically indistinguishable from the minimum point
# That is, the MSE - 1 standard deviation of the error (or squared error)  <= the MSE of the minimum point
