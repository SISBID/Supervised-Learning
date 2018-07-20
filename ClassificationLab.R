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

# Linear Discriminant Analysis

library(MASS)
lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=train)
lda.fit
plot(lda.fit)
lda.pred=predict(lda.fit, Smarket.2005)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,Direction.2005)
mean(lda.class==Direction.2005)
sum(lda.pred$posterior[,1]>=.5)
sum(lda.pred$posterior[,1]<.5)
lda.pred$posterior[1:20,1]
lda.class[1:20]
sum(lda.pred$posterior[,1]>.9)
hist(lda.pred$posterior)

# Quadratic Discriminant Analysis

qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.fit
qda.class=predict(qda.fit,Smarket.2005)$class
table(qda.class,Direction.2005)
mean(qda.class==Direction.2005)

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


