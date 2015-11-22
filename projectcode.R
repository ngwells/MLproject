setwd("C:/Users/nwells/Documents/COURSERA/MachineLearningProject")
set.seed(50)
train<-read.csv("pml-training.csv",h=T)
test<-read.csv("pml-testing.csv",h=T)

#see the data and NA's
for (i in 1:ncol(train))
{
print(paste(names(train)[i],sum(is.na(train[,i]))))
}

##change the blanks to NA
train[train==""]<-NA
##only keep data with less than 10% blank/NA
newtrain<-train[colSums(is.na(train))/dim(train)[1]<.1]
#remove features with unqiue values
newtrain2<-newtrain[,-c(1:5)]

newtrain2$new_window<-as.numeric(newtrain2$new_window)

library(RWeka)

library(party)
library(partykit)


model1<-J48(factor(newtrain2[,55])~.,data=newtrain2[,-55])
model1cv<-evaluate_Weka_classifier(model1,numFolds = 10)
summary(model1)
model1cv
datanames<-names(newtrain2[,-55])

test2<-test[,datanames]
test2[,1]<-as.numeric((test2[,1]))

names(test)
results<-predict(model1,newdata=test2)
table(test$problem_id,results)

for ( i in 2:(ncol(newtrain2)-1))
{
	newtrain2[,i]<-as.numeric(as.character(newtrain2[,i]))
	
}

##############3
#PCA
###################


pca1<-prcomp(newtrain2[,-55])

newpcadata<-data.frame(pca1$x[,1:15])

newdata<-cbind(newpcadata,newtrain2[,"classe"])

newdata<-as.data.frame(newdata)


##PCA j48
model1<-J48(factor(newdata[,16])~.,data=newdata[,-16],,control = Weka_control(R=T))
#,control = Weka_control(C=.25,M=3) # add parameters
model1cv<-evaluate_Weka_classifier(model1,numFolds = 10)
summary(model1)
model1cv
##################33
#Random Forest

library(randomForest)
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
set.seed(30)
#normalized data
class(newdata<-randomForest(factor(newtrain2[,55])~.,data=newtrain2[,-55],importance=TRUE,ntree=2000)
varImpPlot(fsmodel,cex = 0.9, pch = 15,color = "brown", lcolor = "blue",bg="black",type=1,main="ALL Features")
testfs<-importance(fsmodel)
testfs<-as.data.frame(testfs)
testfs<-testfs[with(testfs, order(-MeanDecreaseAccuracy)), ]
varsal120<-rownames(testfs)[1:20]


predRF<-predict(fsmodel,newdata=test2)
table(predRF,test2[,55])



setwd("C:/Users/nwells/Documents/COURSERA/MachineLearningProject")
library(knitr)
start<-Sys.time()

knit2html("MLproject.Rmd","C:/Users/nwells/Documents/Coursera/MachineLearningProject/index.md")
end<-Sys.time()

end-start

