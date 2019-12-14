# ML-Library-1
PROPOSAL FOR “ML LIBRARY” PROJECT IN MACHINE LEARNING 
Table of Contents
1.Description
2.Features
        a) Linear Regression
        b) Logistic Regression
        c) K-Means Clustering
        d) Neural Networks
        e) Some other features
3.Technology stack
4.Implementation Details
5.Week Wise Timeline
          a) Week 1
          b) Week 2
          c) Week 3
          d) Week 4
1.DESCRIPTION
The project aims to develop a machine learning library consisting of various machine learning algorithms and modules for basic mathematical function. The algorithm which would be included are beginner machine learning algorithms like linear regression, logistic regression, K-Means clustering, etc. and more if time permits.
2.FEATURES
a) Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.
Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression.
While training the model we are provided with a training dataset with labelled output using which we produce the best fit for our dataset. When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values.
θ1: intercept
θ2: coefficient of x
Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.
For producing the best fit, we have to minimize the cost function(J). Cost function(J) of Linear Regression is the Root Mean Squared Error (RMSE) between predicted y value (pred) and true y value (y).
To update θ1 and θ2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively updating the values, reaching minimum cost.
b) Logistic Regression
Logistic Regression is Classification algorithm commonly used in Machine Learning. It allows categorizing data into discrete classes by learning the relationship from a given set of labelled data. It learns a linear relationship from the given dataset and then introduces a non-linearity in the form of the Sigmoid function.
In case of Logistic regression, the hypothesis is the Sigmoid of a straight line, i.e, h(x)=sig(w*x + b) where sig(x)=1/(1+exp(-z));
Where the vector ‘W’ represents the Weights and the scalar ‘b’ represents the Bias of the model.
The range of the Sigmoid function is (0, 1) which means that the resultant values are in between 0 and 1. This property of Sigmoid function makes it a really good choice of Activation Function for Binary Classification. Also for z = 0, Sigmoid(z) = 0.5 which is the midpoint of the range of Sigmoid function.
Just like Linear Regression, we need to find the optimal values of w and b for which the cost function J is minimum.

c) K-Means Clustering
We are given a data set of items, with certain features, and values for these features (like a vector). The task is to categorize those items into groups. To achieve this, we will use the k-Means algorithm; an unsupervised learning algorithm.
d)Neural Networks
A neural network is a series of algorithms that endeavours to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria. 
e) 
 All mathematical functions that are needed in ML ( Machine learning ) algorithms will be included.
Test data will be provided to test if all the algorithms are working properly.
The library would be accessible to everyone through a platform on which I will be uploading my project( I will know about this by asking my mentor )
3) Technology stack
a) NumPy and Pandas library 
b) Git and GitHub
c) Jupiter notebook
4) Implementation details
I will write all the algorithms from scratch by implementing the mathematical functions in form of code. For writing my code I will use Jupiter notebook. The project modules would be stored in a package which will be on a Git Repository .I will write different modules for different functions , the modules of ML algorithms will use mathematical function from the mathematics sub package in the home directory. 
5) WEEK WISE TIMELINE
a) Week 1
In week 1 I would complete the ML Coursera course(Currently in  9Th Week) , complete NumPy(Just started) , get used to using Python for ML(as in the course octave is used), get fluent with making modules and packages and at the end will start committing to the repository with setup.py files and other home directory files .I will contact the mentor whenever I find difficulty in proceeding.
b) Week 2
In week 2 I will try to Implement linear and logistic regression model and writing mathematical functions in form of code. Through the week 1 I will continue practice ng NumPy and python to get used to them. At the end of the week I wish to have written the above two algorithms. 

c) Week 3
In this week I will learn KNN model and write its code. After that I will be implementing and testing one of the algorithms of phase 2 that is K-Means clustering. In this week I will be learning thoroughly about neural networks and its applications.
d) Week 4
After thoroughly completing the remaining course and left out phase 2 algorithms I will complete all the remaining work. After completing all the work I will clean up the code, add extra comments where needed and do further task after the instruction of mentor.
