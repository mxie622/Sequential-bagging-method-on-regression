
library(lpSolve)
library(MASS)
library(rpart)
library(mlbench)
library(datasets)
library(e1071)
library(ISLR)
attach(Carseats)
library(randomForest)
library(tree)
library(class)
library(dplyr)

cores <- detectCores()
options(mc.cores=cores)

# Loading dataset. We aim to predict if the person is from US or not
rt.df <- Carseats
# Data transforming
rt.df$Urban <- ifelse(rt.df$Urban =="Yes", 1, 0) 
rt.df$ShelveLoc <- ifelse(rt.df$ShelveLoc == "Bad", 0, ifelse(rt.df$ShelveLoc == "Medium", 1, 2))

data.train <- rt.df[1 : (length(rt.df[, 1]) / 3), ]
data.test <- rt.df[-(1 : (length(rt.df[, 1]) / 3)),]

sapply(rt.df, class)

# Parameters to be tuned
Nsim <- length(rt.df[,1])  #  400

maxit <- length(data.train[, 1]) # input data size#  V3
index <- 1 : maxit
reps <- 50  #  used for replicates
res <- round(maxit / 1.5)  #  used for resamplings
n.predictors <-  length(data.train[1,]) - 1 #  The number of predictors
response <- n.predictors + 1  #  The column of response


y = "y1"
data.train = data.train[,c(which(colnames(data.train) != y), which(colnames(data.train) == y))]  
data.test = data.test[,c(which(colnames(data.test) != y), which(colnames(data.test)==y))]  

mingzi <- paste("x", 1 : length(data.train), sep = "")
mingzi[length(mingzi)] <- "y1"
colnames(data.train) <- mingzi
colnames(data.test) <- mingzi
n.predictors = ncol(data.train) - 1
maxit <- nrow(data.train)  #  input data size# 
index <- 1 : maxit
response <- n.predictors + 1  #  The column position of response

splitting = seq(2, 3, by = 0.01)
result.compare.knn = list()

##### SQB on K-Nearest Neighbour
method <- function(data.train, index){
  store <- double(maxit)
  for (i in index){
    j <- index[i]
    
    subindex1 <- sample((1 : maxit)[-j], res, replace=F)
    bootstrap.sample1 <- data.train[subindex1, ]
    fit1.step2.knn1 <- knn(bootstrap.sample1[, -response], 
                           data.train[j, -response], 
                           bootstrap.sample1[, response], k=3, prob=T)
    store[i] <- fit1.step2.knn1
  }
  store
}

v.knn <- replicate(reps, method(data.train=data.train, index=index))
v.knn <- v.knn - 1
each.mean.knn <- rowMeans(v.knn)
new.class.knn100 <- replicate(reps, rbinom(length(each.mean.knn), 1, each.mean.knn))

training.newL <- cbind(data.train[, -length(rt.df)], new.class.knn100)

ifelse(data.test[, response] == "No", 0, 1)
df.knn <- matrix(0, nrow=nrow(data.test), ncol=reps)

for  (i in (n.predictors + 1) : (reps + n.predictors)) {
  fit.knn <- knn(training.newL[, 1 : n.predictors], data.test[, 1 : n.predictors],
                 training.newL[, i], k=3)
  df.knn[, i - n.predictors] <- fit.knn
}

df.knn <- df.knn - 1
prob.knn100 <- apply(df.knn, 1, mean)
final.class.knn <- ifelse(prob.knn100 > 0.5, "Yes", "No")

tab.knn <- table(final.class.knn, data.test[, response])

pe4.knn <- sum(diag(tab.knn)) / length(fit.knn)
# conventional
fit4 <- knn(data.train[, -response], data.test[, -response], data.train[, response], k=3)  #  knn
pred4 <- table(fit4, data.test[,response])
tab4  <- pred4
pe4 <- sum(diag(tab4)) / sum(tab4)
# Compare SQB_knn and knn where k = 3

c(pe4.knn, pe4)



##### SQB on Random Forest
formula = y1 ~.
method <- function(formula, data.train, index) {
  w <- double(maxit)
  for (i in 1 : maxit) {
    j <- index[i]
    subindex1 <- sample((1 : maxit)[-j], res, replace=F)
    bootstrap.sample1 <- data.train[subindex1, ]
    fit1.step2.rf1 <- randomForest(formula = formula, data = bootstrap.sample1, ntree=100) #randomForest
    pred1.RF <- predict(fit1.step2.rf1, newdata=data.train[j, -response])
    pred1.RF <- as.numeric(pred1.RF)
    w[i] <- pred1.RF
  }
  w
}

RF100 <- replicate(reps, method(formula, data.train, index))
v.RF100 <- RF100 - 1
each.mean.RF <- rowMeans(v.RF100)
new.class.RF100 <- replicate(reps, rbinom(length(each.mean.RF), 1, each.mean.RF))

for (i in 1 : reps) {
  new.class.RF100[, i] <- as.factor(new.class.RF100[, i])
}
new.class.RF100 <- ifelse(new.class.RF100==2, "Yes", "No")
training.newL <- cbind(data.train[, -response], new.class.RF100)

df.RF <- matrix(0, nrow=nrow(data.test), ncol=reps)
for (i in (n.predictors + 1) : (reps + n.predictors)) {
  gongshi.KDE <- as.formula(paste("training.newL[, i] ~",
                                  paste(attr(terms.formula(formula, data = data.train), "term.labels"), 
                                        sep = "", collapse = "+")))
  new.RF.fit <- randomForest(gongshi.KDE, training.newL)
  df.RF[, i - n.predictors] <- predict(new.RF.fit, newdata=data.test)
}
df.RF <- df.RF - 1

probs.RF100 <- apply(df.RF, 1, mean)
final.class.RF <- ifelse(probs.RF100 > 0.5, "Yes", "No")

tab2.RF <- table(final.class.RF, data.test$y1)
pe2.RF <- sum(diag(tab2.RF)) / length(final.class.RF)
# conventional
fit2 <- randomForest(y1 ~ ., data=data.train) #  Random forest
pred2 <- predict(fit2, newdata=data.test)
tab2 <- table(pred2, data.test[, response])
pe2 <- sum(diag(tab2)) / length(pred2)

c(pe2.RF, pe2)


##### SQB on naiveBayes

formula = y1~.
method <- function(data.train, index) {
  w <- double(maxit)
  for (i in index) {
    j <- index[i]
    subindex1 <- sample((1 : maxit)[-j], res, replace=F)
    bootstrap.sample1 <- data.train[subindex1, ]
    fit1.step2.NB1 <- naiveBayes(formula = formula, data=bootstrap.sample1) #naive bayes
    pred1.NB <- predict(fit1.step2.NB1, newdata=data.train[j, -response])
    pred1.NB <- as.numeric(pred1.NB )
    w[i] <- pred1.NB
  }
  w
}

v.NB <- replicate(reps, method(data.train=data.train, index=index))
v.NB100 <- v.NB - 1
each.mean.NB <- rowMeans(v.NB100)
new.class.NB100 <- replicate(reps, rbinom(length(each.mean.NB), 1, each.mean.NB))
new.class.NB100 <- ifelse(new.class.NB100==1, "Yes", "No")
training.newL <- cbind(data.train[, -response], new.class.NB100)

df.NB <- matrix(0, nrow=nrow(data.test), ncol=reps)
for (i in (n.predictors + 1) : (reps + n.predictors)) {
  gongshi.KDE <- as.formula(paste("training.newL[, i] ~",
                                  paste(attr(terms.formula(formula, data = data.train), "term.labels"), 
                                        sep = "", collapse = "+")))
  
  new.NB.fit <- naiveBayes(gongshi.KDE, data=training.newL)
  df.NB[, (i - n.predictors)] <- predict(new.NB.fit, newdata=data.test)
}
df.NB <- df.NB - 1

probs.NB100 <- apply(df.NB, 1, mean)
final.class.NB <- ifelse(probs.NB100 > 0.5, "Yes", "No")

tab3.NB <- table(final.class.NB, data.test[, response])

pe3.NB <- sum(diag(tab3.NB)) / length(final.class.NB)

fit3 <- naiveBayes(formula, data=data.train) #  naiveBayes
pred3 <- predict(fit3, newdata=data.test)
tab3 <- table(pred3, data.test[, response])
pe3 <- sum(diag(tab3)) / length(pred3)
c(pe3.NB, pe3)


######### Conventional Bagging
splitting = seq(2, 3, by = 0.01)
bagging = list()
result.bag = list()
bag.pred = list()
conventional.bag = c()
average_table_result = double(reps)
for (i in 1 : reps){
  bagging[[i]] = data.train[sample(1:nrow(data.train), round(nrow(data.train) * 0.5), replace=T), ]
  
   result.bag[[i]] = knn(bagging[[i]][, -response], 
                         data.test[, -response],
                         bagging[[i]][, response], k = 3) # knn
  # result.bag[[i]] = randomForest(formula = y1 ~ ., data = bagging[[i]], ntree = 100) # RF
  # result.bag[[i]] = naiveBayes(formula = y1 ~ ., data = bagging[[i]]) # NB
  
  # bag.pred[[i]] <- table(predict(result.bag[[i]], newdata=data.test),
  #                       data.test[, response]) # RF & NB
  
  bag.pred[[i]] <- table(result.bag[[i]], data.test[, response]) # knn
  average_table_result[i] = sum(diag(bag.pred[[i]])) / sum(bag.pred[[i]])
}
bagging_result = mean(average_table_result)
bagging_result




