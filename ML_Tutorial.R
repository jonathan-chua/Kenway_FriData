###### FriData Machine Learning Tutorial

# Load Packages
library(kknn) 
#library(tree)
#library(randomForest)
#library(rpart)

# Load data
# Preload
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

# Store max and min to format chart axes
max.x <- max(caldata$medianIncome)
min.x <- min(caldata$medianIncome)
max.y <- max(caldata$logMedVal)
min.y <- min(caldata$logMedVal)

# To properly test accuracy, we need to set up train, test, and validation sets
# Train sets are used to develop the models
# Test sets are used for model selection and tuning
# Validation sets are the final proving ground for a model
# We'll go with 60% train, 20% test, 20% validation

# Preload

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
cal.cv <- rbind(cal.test, cal.train)

# Run Linear Regression

# Preload
lr.cal <- lm(logMedVal ~ medianIncome, data=cal.cv)

abline(lr.cal, col="red", lwd=2)

# Ok fit, there are issues with the data (top-coded / upper bounded) but it fits the general trend

summary(lr.cal)
# Very significant

lr.cal$coefficients
# For each unit increase of Median Income, observe a 0.196 increase in Log Median Value

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
# Preload
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

# How can we improve this? Can we bring in more variables? Conceptually, things like
# Latitude and Longitude will have significant effects on prices...

# Re-plot
plot(caldata$medianIncome, caldata$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgray",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

# Run Linear Regression
# Preload
# NOTE "." means all variables
lr.calFull <- lm(logMedVal ~ . , data=cal.train)

pred.calFull <- cal.val[order(cal.val$medianIncome),]

pred.calFull <- cbind(pred.calFull, yhat=predict(lr.calFull, newdata=pred.calFull))

line.calFull <- setNames(aggregate(pred.calFull$yhat, by=list(pred.calFull$medianIncome), FUN="mean"), c('medianIncome', 'logMedIncome'))

lines(line.calFull$medianIncome, line.calFull$logMedIncome, col="red", lwd=2)

# This line appears to catch more of the variance BUT it's also variance chasing

# Get predictions for the data (skipping to validation set because no tuning)
lrFull.yhat <- predict(lr.calFull, newdata=cal.val)

# Calculate MSE
lrFull.mse <- calc.mse(lrFull.yhat, cal.val$logMedVal)

print(paste("Linear Regression w/ All Variables MSE = ", round(lrFull.mse, 5), sep=""))
# Performs better than original regression with one variable

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="Linear Regression, All Variables", mse=lrFull.mse))

# To capture some of the non-linear, complex relationships and improve predictions
# use Machine Learning

##### k-NN Regression

# Replot
plot(caldata$medianIncome, caldata$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgrey",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

# Plot initial effort
# Create sample Median Incomes
sample.income <- cal.val[order(cal.val$medianIncome),]

# Run for Median Income only w/ k=25
# Preload
knn.calInc <- kknn(logMedVal ~ medianIncome, train=cal.cv, test=sample.income, k=25, kernel="rectangular")

lines(sample.income$medianIncome, knn.calInc$fitted.values, col="blue", lwd=1)

# Get Accuracy for Median Income and full models
knnInc.yhat <- knn.calInc$fitted.values

knnInc.mse <- calc.mse(knnInc.yhat, cal.val$logMedVal)

print(paste("25-NN w/ Median Income = ", round(knnInc.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="25-NN, Median Income Only", mse=knnInc.mse))
# Actually worse than Linear Regression
# This is because it was an untuned version, i.e. we randomly selected k instead of using
# Machine Learning Cross Validation to find the optimal one

# k-NN Illustration
##### ***** Please use this, I'm very proud of it
# Function to calculate Euclidean distance
# Preload
calc.dist <- function(y, yhat) {
  return( (abs(yhat-y)^2)^1/2 )
}

viz <- data.frame(medianIncome=sample.income$medianIncome, logMedVal=sample.income$logMedVal, yhat=knn.calInc$fitted.values)
n <- nrow(viz)
keep <- seq(1,n,200)

viz <- viz[keep,]

n.viz <- nrow(viz)

viz <- cbind(viz, n=1:n.viz)

for (iter in 1:n.viz) {
  x <- viz[iter, "medianIncome"]
  row <- viz[iter, "n"]
  dist <- data.frame(
    medianIncome=cal.train$medianIncome, 
    logMedVal=cal.train$logMedVal, 
    distance=calc.dist(cal.train$medianIncome, x)
  )
  
  plot(
    sample.income$medianIncome,
    sample.income$logMedVal,
    pch=16,
    col="darkgray",
    xlab="Median Income",
    ylab="ln(Median Value)",
    main="ln(Median Value) by Median Income"
  )
  
  lines(sample.income$medianIncome, knn.calInc$fitted.values, col="red")
  
  neighbors <- dist[order(dist$distance),]
  points(neighbors$medianIncome[1:25], neighbors$logMedVal[1:25], col="blue", pch=16)
  points(viz$medianIncome[row], viz$yhat[row], col="black", pch=4, lwd=4, cex=2)
  #readline("go?")
  Sys.sleep(.4)
}

# Full data
full.test <- cal.test[order(cal.test$medianIncome), ]

knn.calFull <- kknn(logMedVal ~ . , train=cal.train, test=full.test, k=25, kernel="rectangular", distance=2)

# Plot full k-NN model

# Replot
plot(caldata$medianIncome, caldata$logMedVal,
     # Main label
     main="log(Median Value) by Median Income",
     # Dot color
     col="darkgrey",
     # X-Axis
     xlab="Median Income",
     # Y-Axis
     ylab="log(Median Value)")

lines(full.test$medianIncome, knn.calFull$fitted.values, col="green", lwd=1)

lines(sample.income$medianIncome, knn.calInc$fitted.values, col="blue", lwd=1)

# We see the new model is working to capture more variance

legend("bottomright", legend=c("Median Income", "Full Model"), fill=c("blue", "green"))  

knnFull.yhat <- knn.calFull$fitted.values

knnFull.mse <- calc.mse(knnFull.yhat, full.test$logMedVal)

print(paste("25-NN w/ All Fields = ", round(knnFull.mse, 5), sep=""))

# Add to performance table
model.perf <- rbind(model.perf, data.frame(model="25-NN, All Fields", mse=knnFull.mse))
# Significantly better performance than random k-nn with income, better than both regressions

# Notice how the neighbors changed
calc.dist_2 <- function(matrix, vector) {
  d <- ncol(matrix)
  dist <- matrix(NA, ncol=0, nrow=nrow(matrix))
  for (z in 1:d) {
    v <- as.numeric(vector[z])
    dist <- cbind(dist, data.frame(d=abs(matrix[,z]-v)^2))
  }
  return(rowSums(dist)^(1/2))
}

viz <- data.frame(full.test, yhat=knn.calFull$fitted.values)
n <- nrow(viz)
keep <- seq(1,n,200)

viz <- viz[keep,]

n.viz <- nrow(viz)

viz <- cbind(viz, n=1:n.viz)

for (iter in 1:n.viz) {
  x <- viz[iter, 1:9]
  row <- viz[iter, "n"]
  dist <- data.frame(
    medianIncome=cal.train$medianIncome, 
    logMedVal=cal.train$logMedVal, 
    distance=calc.dist_2(cal.train[,1:9], x)
  )
  
  plot(
    full.test$medianIncome,
    full.test$logMedVal,
    pch=16,
    col="darkgray",
    xlab="Median Income",
    ylab="ln(Median Value)",
    main="ln(Median Value) by Median Income",
    xlim=c(0,max.x),
    ylim=c(min.y*0.9,max.y)
  )
  
  lines(full.test$medianIncome, knn.calFull$fitted.values, col="red", lwd=0.5)
  
  neighbors <- dist[order(dist$distance),]
  points(neighbors$medianIncome[1:25], neighbors$logMedVal[1:25], col="blue", pch=16)
  points(viz$medianIncome[row], viz$yhat[row], col="black", pch=4, lwd=4, cex=2)
  #readline("go?")
  Sys.sleep(.4)
}

# We capture more of the variance by factoring in other variables to our "nearness" metric

# To improve the model, let's look at tuning for k with all variables

# Show how increasing k decreases complexity & model-fitting
# Show charts in a 2x2 matrix
par(mfrow=c(2,2))
# Going to test k from 1, 10, 20, ..., 100
nn.test <- c(1,seq(10,100,10))
nn.col <- length(nn.test)

# Loop through the numbers, run the k-NN regression, plot the line
for (i in 1:nn.col) {
  nn <- nn.test[i]
  knn.fitting <- kknn(logMedVal ~ medianIncome, train=cal.train, test=full.test, k=nn, kernel = "rectangular")
  m.fitting <- cbind(data.frame(knn.fitting$fitted.values))
  plot(full.test$medianIncome, full.test$logMedVal, xlab="Median Income", ylab="log(Median Value)", col="darkgray", main=paste("k = ", nn, sep=""))
  lines(full.test$medianIncome, knn.fitting$fitted.values, col=i, lwd=2)
}

par(mfrow=c(1,1))

# As you can see, the lines get tighter as k increases;
# a tighter line indicates a more simple model, i.e. a model that is chasing less noise.
# This makes for a model that is more robust because it's not as sensitive to new variance.
# If we ran a k-NN regression where k=n_observations, we would iget a linear regression

# Tuning - picking the best k to optimize prediction
# This is where train & test are used, validation is left out so that it is completely independent of our tuning

# Capture nn performance

# Run model for k from 1 to 100, MSE for each value
# Preload
nn.mse <- matrix(NA, ncol=0, nrow=0)
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

# What this tells us is that based on cross-validating each k with our train and test sets,
# k=11 is the best value to use.
# We need to confirm this against the validation set and compare that with our old models to
# pick the "best" one for implementation

knn.optimal <- kknn(logMedVal ~ . , train=cal.train, test=cal.val, k=min.k$k, kernel="rectangular")
optimal.train <- calc.mse(knn.optimal$fitted.values, cal.test$logMedVal)

print(paste("k-NN, Full Model tuning on k = ", round(optimal.train, 5), sep=""))

model.perf <- rbind(model.perf, data.frame(model="k-NN, Full MOdel tuning on k", mse=optimal.train))

# This does terribly
# Why? Looking at the training sample we get:
nn.mse[1:15,]
# k=11 has a MSE of 0.086, the best performing thus far! However, this model is not robust.
# It performed great in the train vs. test environment, but we need it to perform in the real
# world (simulated by the validation set).
# How to achieve this? Using k-folds cross validation

# 10-folds to use SD
# k-folds CV runs the validation multiple times against different sets of the data
# Predictions are then based on the average of the estimates for a specific k
# Since we have multiple samples, we can find simpler models that are not statistically
# distinguishable from the optimal k.

# SD Function
calc.sde <- function(yhat, y) {
  e <- yhat - y
  se <- e^2
  return(sd(se))
}

# Number of folds
f <- 10
# Total rows
n <- nrow(cal.cv)
# Records per fold
l.fold <- round(n/f,0)

# 10-fold k-NN
# PRELOAD THIS TAKES FOREVER
fold.nnMSE <- matrix(NA, ncol=0, nrow=0)

for (folds in 1:f) {
  # Identify 1/10 of the data as a test set
  f.start <- if (folds==1) {1} else {f.start+l.fold}
  f.end <- if (folds==1) {f.start+l.fold-1} else {f.end+l.fold}
  i.fold <- f.start:f.end
  
  # Mark the training set as the remaining 9/10
  f.train <- cal.cv[-i.fold,]
  # Mark the test set
  f.test <- cal.cv[i.fold,]
  
  for (nn in 1:100) {
    f.knn <- kknn(logMedVal ~ . , train=f.train, test=f.test, k=nn, kernel="rectangular")
    f.mse <- calc.mse(f.knn$fitted.values, f.test$logMedVal)
    fold.nnMSE <- rbind(fold.nnMSE, data.frame(k=nn, mse=f.mse, fold=folds))
  }
}

# Get the average and SD for each k

fold.mse <- data.frame(setNames(aggregate(fold.nnMSE$mse, by=list(fold.nnMSE$k), FUN="mean"), c("k","mse")))
fold.sde <- data.frame(setNames(aggregate(fold.nnMSE$mse, by=list(fold.nnMSE$k), FUN="sd"), c("k","sde")))  


# Plot the performance
plot(fold.mse$k, fold.mse$mse, type="l", col="darkgray", lwd=2,
     main="MSE by k",
     xlab="k",
     ylab="MSE")

points(fold.mse$k, fold.mse$mse, col="darkgray", pch=16)

fmin.k <- fold.mse[which.min(fold.mse$mse), ]

points(fmin.k$k, fmin.k$mse, pch=16, col="red")

print(paste("MSE is minimized at k = ", fmin.k$k, "; MSE = ", round(fmin.k$mse, 4), sep=""))

abline(h=fmin.k$mse, lty=2, lwd=1, col="black")

# Factor in SD

segments(fold.sde$k, fold.mse$mse+fold.sde$sde, fold.sde$k, fold.mse$mse-fold.sde$sde, col="darkgray")

f.optimal <- data.frame(fold.mse, mse_sde=fold.mse$mse-fold.sde$sde)

f.optimal2 <- f.optimal[f.optimal$mse_sde <= fmin.k$mse, ]

f.optimal3 <- f.optimal2[which.max(f.optimal2$k),]

points(f.optimal3$k, f.optimal3$mse, pch=18, col="darkgreen", cex=1.5)

print(paste("While k = ", fmin.k$k, " minimizes MSE, it is not statistically distinguishable from k = ", f.optimal3$k,". This will be used for the final calculation. It has a MSE = ", round(f.optimal3$mse, 4), sep=""))

# Compare to validation set
knn.fullFinal <- kknn(logMedVal ~ ., train=cal.cv, test=cal.val, k=f.optimal3$k, kernel="rectangular")

knnFinal.mse <- calc.mse(knn.fullFinal$fitted.values, cal.val$logMedVal)

print(paste("k-NN, Full MOdel tuning on k w/ 10-fold CV = ", round(knnFinal.mse, 5), sep=""))

model.perf <- rbind(model.perf, data.frame(model="k-NN, Full MOdel tuning on k w/ 10-fold CV", mse=knnFinal.mse))

# This one is the best performing of our tests

# Show how it works
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

# View performance
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

# It looks like we're pretty accurate; the points are fairly close and evenly distributed

# Residual analysis
plot(knn.fullFinal$fitted.values, knn.fullFinal$fitted.values-cal.val$logMedVal, col="darkgray", xlab="Y_hat", ylab="Residuals", main="Predictions vs. Residuals")
abline(h=0, lty=2, lwd=2, col="blue")

# Outside of the top-coding, our predictions also look homoskedastic