# CSE474proj3


CSE474/574 Introduction to Machine Learning
Programming Assignment 3
Classification and Regression
Due Date: May 1st 2015
Maximum Score: 100
Note
A zipped file containing skeleton Python script files and data is provided. Note that for each problem, you
need to write code in the specified function withing the Python script file. For logistic regression, do not
use any Python libraries/toolboxes, built-in functions, or external tools/libraries that directly
perform the learning or prediction.. Using any external code will result in 0 points for that problem.
Evaluation
We will evaluate your code by executing script.py file, which will internally call the problem specific
functions. Also submit an assignment report (pdf file) summarizing your findings. In the problem statements
below, the portions under REPORT heading need to be discussed in the assignment report.
1 Introduction
In this assignment, we will extend the first programming assignment in solving the problem of handwritten
digit classification. In particular, your task is to implement Logistic Regression and use the Support Vector
Machine tool in sklearn.svm.SVM to classify hand-written digit images and compare the performance of
these methods.
To get started with the exercise, you will need to download the supporting files and unzip its contents to
the directory you want to complete this assignment.
1.1 Datasets
In this assignment, we still use the same data set of first programming assignment - MNIST. In the script
file provided to you, we have implemented a function preprocess() with preprocessing steps (apply feature
selection and feature normalization) and divide data set into 3 parts: training set, validation set and testing
set.
2 Your tasks
• Implement Logistic Regression and give the prediction results.
• Use Support Vector Machine (SVM) toolbox to perform classification.
• Write a report to explain the experimental results with these 2 methods.
1
2.1 Logistic Regression
2.1.1 Formula derivation
Suppose that a given vector x ∈ R
D is any input vector and we want to classify x into correct class C1 or
C2. In Logistic Regression, the posterior probability of class C1 can be written as follow:
y = P(C1|x) = σ(wT x + w0)
where w ∈ R
D is the weight vector.
For simplicity, we will denote x = [1, x1, x2, · · · , xD] and w = [w0, w1, w2, · · · , wD]. With this new notation,
the posterior probability of class C1 can be rewritten as follow:
P(C1|x) = σ(wT x) (1)
And posterior probability of class C2 is:
P(C2|x) = 1 − P(C1|x)
We now consider the data set {x1, x2, · · · , xN } and corresponding label {t1, t2, · · · , tN } where
ti =

1 if xi ∈ C1
0 if xi ∈ C2
for i = 1, 2, · · · , N.
With this data set, the likelihood function can be written as follow:
p(t|w) = Y
N
n=1
y
tn
n
(1 − yn)
1−tn
where yn = σ(wT xn) for n = 1, 2, · · · , N.
We also define the error function by taking the negative logarithm of the log likelihood, we gives the crossentropy
error function of the form:
E(w) = − ln p(t|w) = −
X
N
n=1
{tn ln yn + (1 − tn) ln(1 − yn)} (2)
The gradient of error function with respect to w can be obtained as follow:
∇E(w) = X
N
n=1
(yn − tn)xn (3)
Up to this point, we can use again - gradient descent - to find the optimal weight wˆ to minimize the error
function with the formula:
wnew = wold − γ∇E(wold) (4)
2.1.2 Implementation
You are asked to implement Logistic Regression to classify hand-written digit images into correct corresponding
labels. In particular, you have to build 10 binary-classifiers (one for each class) to classify that class from
all other classes. In order to implement Logistic Regression, you have to complete function blrObjFunction()
provided in the base code (script.py). The input of blrObjFunction.m includes 3 parameters:
• X is a data matrix where each row contains a feature vector in original coordinate (not including the
bias 1 at the beginning of vector). In other words, X ∈ R
N×D. So you have to add the bias into
each feature vector inside this function. In order to guarantee the consistency in the code and utilize
automatic grading, please add the bias at the beginning of feature vector instead of the end.
2
• w is a column vector representing the parameters of Logistic Regression. Size of w is (D + 1) × 1.
• t is a column vector representing the labels of corresponding feature vectors in data matrix X. Each
entry in this vector is either 1 or 0 to represent whether the feature vector belongs to a class Ck or not
(k = 1, 2, · · · , 10). Size of t is N × 1 where N is the number of rows of X.
Function blrObjFunction() has 2 outputs:
• error is a scalar value which is the result of computing equation (2)
• error grad is a column vector of size (D + 1) × 1 which represents the gradient of error function
obtained by using equation (3).
For prediction using Logistic Regression, given 10 weights vector of 10 classes, we need to classify a feature
vector into a certain class. In order to do so, given a feature vector x, we need to compute the posterior
probability P(Ck|x) and the decision rule is to assign x to class Ck that maximizes P(Ck|x). In particular,
you have to compltete the function blrPredict() which returns the predicted label for each feature vector.
Concretely, the input of blrPredict() includes 2 parameters:
• Similar to function blrObjFunction(), X is also a data matrix where each row contains a feature vector
in original coordinate (not including the bias 1 at the beginning of vector). In other words, X has size
N × D. In order to guarantee the consistency in the code and utilize automatic grading, please add
the bias at the beginning of feature vector instead of the end.
• W is a matrix where each column is a weight vector of classifier k. Concretely, W has size (D + 1)×K
where K = 10 is the number of classifiers.
The output of function blrPredict() is a column vector label which has size N × 1, in which a feature vector
is classify to digit k will have label k + 1.
2.2 Support Vector Machines
In this part of assignment, you are asked to use the Support Vector Machine tool in sklearn.svm.SVM to
perform classification on our data set. The details about the tool are provided here: http://scikit-learn.
org/stable/modules/generated/sklearn.svm.SVC.html.
Your task is to fill the code in Support Vector Machine section of script.py to learn the SVM model and
compute accuracy of prediction with respect to training data, validation data and testing using the following
parameters:
• Using linear kernel (all other parameters are kept default).
• Using radial basis function with value of gamma setting to 1 (all other parameters are kept default).
• Using radial basis function with value of gamma setting to default (all other parameters are kept
default).
• Using radial basis function with value of gamma setting to default and varying value of C (1, 10, 20, 30, · · · , 100)
and plot the graph of accuracy with respect to values of C in the report.
3 Submission
You are required to submit a single file called proj3.zip using UBLearns.
File proj3.zip must contain 3 files: report, params.pickle and script.py. The params.pickle file should contain
the weight matrix,W, learnt for the logistic regression
• Submit your report in a pdf format. Please indicate the team members, group number, and your course
number on the top of the report.
• The code file should contain all implemented functions. Please do not change the name of the file.
3
Using UBLearns Submission: Continue using the groups that you created for programming
assignment 1. You should submit one solution per group through the groups page. If you want to
change the group, contact the instructors.
Project report: The hard-copy of report will be collected in class at due date. Your report should include
the experimental results you have performed using Logistic Regression and Support Vector Machine.
4 Grading scheme
• Implementation:
X blrObjFunction(): 20 points
X blrPredict(): 20 points
X script.py: 20 points (your code in SVM section)
• Project report: 30 points
• Accuracy of classification methods: 10 points