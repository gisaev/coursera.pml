# Classifying activity types using accelerometer data



```r
require(knitr)
```

```
## Loading required package: knitr
```

```r
require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(FSelector)
```

```
## Loading required package: FSelector
```

```r
require(ggplot2)
opts_chunk$set(echo = TRUE, cache = TRUE, cache.path = "cache/", fig.path = "figure/")
```

# Executive summary
  In this paper i implement a machine learning algorithm which uses a collection of accelerometer data to identify if test subjects executed certain physical activity correctly from a technical standpoint.
  The CFS filter agorithm proposed by Hull is used for theature selection and a random forest algorithm with 5-fold crossvalidation is used in prediction model.
  The resulting algorithm has 98.5% accuracy on test (cross validated) data and has successfully identified 20 out of 20 of test tasks, which is inline with theoretical performance.


# Analysis
  In order to implement the machine learning algorithm one of the most important things to do is feature selection. Since it is hard to have a special insight into what kind of features would be relevent for the task from the theoretical perspective, we need to use some kind of automated feature selection algorithm. Following other research in the field we use Correlation-Based Feature Selection algorithm proposed by Mark A.Hull and implemented in Fselector package available on Cran.
  When we apply a random forest algorithm to the selected features and report results.

First we load the test and training data.

```r
pml.training <- read.csv("pml-training.csv", header=TRUE)
pml.testing <- read.csv("pml-testing.csv", header = TRUE)
pml.training$X <- NULL
pml.testing$X <- NULL
```

Next we select relevant features using Fselector package (ignoring timestamps and some other essentially non-feature data). The following 8 features are beeing selected: pitch_belt, yaw_belt, magnet_belt_z, gyros_arm_x, magnet_arm_x, gyros_dumbbell_y, magnet_dumbbell_y, pitch_forearm.
We use random forest with 5-fold cross validation algorithm as implemented in caret package on the training dataset/



```r
subset <- cfs(classe~.,pml.training[,8:ncol(pml.training)])
f <- as.simple.formula(subset, "classe")
print(f)
```

```
## classe ~ pitch_belt + yaw_belt + magnet_belt_z + gyros_arm_x + 
##     magnet_arm_x + gyros_dumbbell_y + magnet_dumbbell_y + pitch_forearm
## <environment: 0x3a6f846c>
```

Now that we got our features selection, we implement random forest with 5-fold cross validation.

```r
modfit <- train(classe ~ pitch_belt + yaw_belt + magnet_belt_z + gyros_arm_x + magnet_arm_x + gyros_dumbbell_y + magnet_dumbbell_y + pitch_forearm, method="rf", trControl=trainControl(method="cv",number=5), model=FALSE, data=pml.training)
```

The accuracy of the model looks very good on the test, it is around 98.5%:

```r
print(modfit)
```

```
## Random Forest 
## 
## 19622 samples
##   158 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 15697, 15697, 15697, 15698, 15699 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##   2     0.9864948  0.9829171  0.001965215  0.002488269
##   5     0.9836922  0.9793707  0.003933014  0.004981149
##   8     0.9795130  0.9740859  0.003604707  0.004565250
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

The OOB estimate of error  rate is 1.11% and the confusion matrix is presented below:

```r
print(modfit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, model = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 1%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5556    8    4    9    3 0.004301075
## B   26 3713   53    2    3 0.022122728
## C    3   18 3395    5    1 0.007890123
## D    1    1   26 3182    6 0.010572139
## E    0   13    7    7 3580 0.007485445
```

