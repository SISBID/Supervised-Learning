############################################
## SISBID3: Classification
############################################

# The Stock Market Data
library(ISLR)
library(MASS)
library(glmnet)
library(class)
library(e1071)
library(ROCR)

names(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket)
cor(Smarket)
cor(Smarket[,-9])
attach(Smarket)
plot(Volume)

# Logistic Regression
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
summary(glm.fit)
coef(glm.fit)
summary(glm.fit)$coef

glm.probs=predict(glm.fit,type="response")
glm.probs[1:10]
contrasts(Direction)
glm.pred=rep("Down",1250)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction)
(507+145)/1250
mean(glm.pred==Direction)

# separate the data into training and test
train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fit,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
mean(glm.pred!=Direction.2005)

glm.fit=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fit,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
106/(106+76)
predict(glm.fit,newdata=data.frame(Lag1=c(1.2,1.5),Lag2=c(1.1,-0.8)),type="response")

# penalized logistic regression
x=as.matrix(Smarket[,c(2:7)])
y=as.factor(Smarket[,9])

#ridge
glm.ridge=glmnet(x[train,],y[train],family="binomial",alpha=0,lambda=exp(seq(-5,10,0.01)))
plot(glm.ridge,col=c(1:8), xvar="lambda")
legend('bottomright',colnames(x),lty=1,col=c(1:8),bty='n')
cv.ridge=cv.glmnet(x[train,],y[train],family="binomial",alpha=0,lambda=exp(seq(-50,2,0.1)))
plot(cv.ridge)
lambda.min=cv.ridge$lambda.min
ridge.probs=predict(glm.ridge,s=lambda.min,type="response",newx=x[!train,])
ridge.pred=rep("Down",252)
ridge.pred[ridge.probs>.5]="Up"
table(ridge.pred,Direction.2005)
mean(ridge.pred==Direction.2005)

#lasso
glm.lasso=glmnet(x[train,],y[train],family="binomial",alpha=1)
plot(glm.lasso,col=c(1:8))
legend('bottomleft',colnames(x),lty=1,col=c(1:8),bty='n')
cv.lasso=cv.glmnet(x[train,],y[train],family="binomial",alpha=1)
plot(cv.lasso)
lambda.min=cv.lasso$lambda.min
lasso.probs=predict(glm.lasso,s=lambda.min,type="response",newx=x[!train,])
lasso.pred=rep("Down",252)
lasso.pred[lasso.probs>.5]="Up"
table(lasso.pred,Direction.2005)
mean(lasso.pred==Direction.2005)
predict(glm.lasso,s=lambda.min,type="coefficients",newx=x[!train,])

#lasso with other choices of lambda
plot(glm.lasso,col=c(1:8),xvar="lambda")
lasso.probs=predict(glm.lasso,s=exp(-5),type="response",newx=x[!train,])
lasso.pred=rep("Down",252)
lasso.pred[lasso.probs>.5]="Up"
table(lasso.pred,Direction.2005)
mean(lasso.pred==Direction.2005)

# K-Nearest Neighbors
library(class)
train.X=cbind(Lag1,Lag2)[train,]
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)
############################################

############################################
## Another example: Classification with the Heart data
############################################

## Need to load the Heart data, available on github
load("Heart.rda")
attach(Heart)

names(Heart)
dim(Heart)
summary(Heart)
plot(trestbps)

# separate the data into training and test
set.seed(1)
train=sample(1:nrow(Heart), 197)
trndat=Heart[train,]
tstdat=Heart[-train,]
dim(trndat)
dim(tstdat)

# Logistic Regression
HD.tst=HD[-train]
glm.fit=glm(HD~.,data=Heart,family=binomial,subset=train)
glm.probs=predict(glm.fit,tstdat,type="response")
glm.pred=numeric(100)
glm.pred[glm.probs>.5]=1
table(glm.pred,HD.tst)
mean(glm.pred==HD.tst)

# penalized logistic regression
x=as.matrix(Heart[,1:13])
y=as.factor(Heart[,14])

#ridge
glm.ridge=glmnet(x[train,],y[train],family="binomial",alpha=0)
plot(glm.ridge,col=c(1:8), xvar="lambda")
legend('topright',colnames(x),lty=1,col=c(1:8),bty='n')
cv.ridge=cv.glmnet(x[train,],y[train],family="binomial",alpha=0)
plot(cv.ridge)
lambda.min=cv.ridge$lambda.min
ridge.probs=predict(glm.ridge,s=lambda.min,type="response",newx=x[-train,])
ridge.pred=rep(0,100)
ridge.pred[ridge.probs>.5]=1
table(ridge.pred,HD.tst)
mean(ridge.pred==HD.tst)

#lasso
glm.lasso=glmnet(x[train,],y[train],family="binomial",alpha=1)
plot(glm.lasso,col=c(1:8), xvar="lambda")
legend('bottomright',colnames(x),lty=1,col=c(1:8),bty='n')
cv.lasso=cv.glmnet(x[train,],y[train],family="binomial",alpha=1)
plot(cv.lasso)
lambda.min=cv.lasso$lambda.min
lasso.probs=predict(glm.lasso,s=lambda.min,type="response",newx=x[-train,])
lasso.pred=rep(0,100)
lasso.pred[ridge.probs>.5]=1
table(lasso.pred,HD.tst)
mean(lasso.pred==HD.tst)


# K-Nearest Neighbors

library(class)
train.X=cbind(ca,thal)[train,]
test.X=cbind(ca,thal)[-train,]
HD.trn=HD[train]
set.seed(1)
knn.pred=knn(train.X,test.X,as.factor(HD.trn),k=1)
table(knn.pred,HD.tst)
mean(knn.pred==HD.tst)
knn.pred=knn(train.X,test.X,HD.trn,k=10)
table(knn.pred,HD.tst)
mean(knn.pred==HD.tst)

# SVM

dat=Heart
dat$HD=as.factor(HD)
svmfit=svm(HD~.,data=dat[train,],kernel="radial",gamma=1,cost=1)
svm.pred=predict(svmfit,newdata=dat[-train,-14])
table(true=dat[-train,14], pred=svm.pred)
mean(dat[-train,14]==svm.pred)

set.seed(1)
tune.out=tune.svm(x=dat[train,-14],y=dat[train,14],kernel="radial",
                  cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4))
summary(tune.out)
svmfit=svm(HD~.,data=dat[train,],kernel="radial",gamma=tune.out$best.parameters$gamma,cost=tune.out$best.parameters$cost)
svm.pred=predict(svmfit,newdata=dat[-train,-14])
table(true=dat[-train,14], pred=svm.pred)
mean(dat[-train,14]==svm.pred)


# ROC Curves

rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

svmfit.opt=svm(y~., data=dat[train,], kernel="radial",gamma=2, cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values
par(mfrow=c(1,2))
rocplot(fitted,dat[train,"y"],main="Training Data")
svmfit.flex=svm(y~., data=dat[train,], kernel="radial",gamma=50, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted,dat[train,"y"],add=T,col="red")
legend('bottomright',c('optimal model','flexible model'),lty=1,col=c(1,2),bty='n')
fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")
fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")

############################################
