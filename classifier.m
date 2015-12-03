%% EXERCISE 2: CLASSIFIER
% * Submitted by: Abhishek Saurabh, Juanjo Rubio and Miguel M?ndez
% * Program: Master's in Artificial Intelligence
% * Course: Introduction to Machine Learning
% * Due date: 29-Nov-2015

%% Data set analysis
diabetes_db = load('diabetes.mat');
%%
% * Which is the cardinality of the train set?
%
cardinality = size(diabetes_db.x,2)
%%
% * Which is the dimensionality of the train set?
%
dimensionality = size(diabetes_db.x,1)
%%
% * Which is the mean value of the train set?
%mean_diabetes = mean2(diabetes_db.x)
mean_diabetes_attr = mean(diabetes_db.x,2)
%%
% * 1) Create a new dataset D1, replacing the NaN values with the mean
% value of the corresponding attribute without considering the missing values.
D1 = diabetes_db.x;
for i=1:size(diabetes_db.x,1)
    attr = diabetes_db.x(i,:);
    nan_pos = isnan(diabetes_db.x(i,:));
    attr_no_nan = ~nan_pos;
    %attr_mean(i) = mean(diabetes_db.x(nan_pos,i));
    D1(i,nan_pos) = mean(diabetes_db.x(i,attr_no_nan));
end
%%
% * 2) Create a new dataset D2, replacing the NaN values with the mean 
% value of the corresponding attribute without considering the missing 
% values conditioned to the class they belong, i.e. replace the missing 
% attribute values of class +1 with the mean of that attribute of the 
% examples of class +1, and the same for the other class.

D2 = preprocessData2(diabetes_db.x,diabetes_db.y);

%%
% * 4) Which are the new mean values of each dataset?

mean_d1 = mean(D1,2)
mean_d2 = mean(D2,2)

%% A simple classifier
%
%%
% * 1) In this model you have to learn the threshold value. Explain how you
% can accomodate this parameter.
%Set the initial parameters
w_in = ones(1,size(D1,1)+1);
maxIter = 200000;
minError = 1e-5;
t = 0.000001;
[ w_D1, costFunction_D1 ] = gradient_descent( D1, diabetes_db.y, maxIter, minError,w_in,t );
[ y_classified_D1 ] = linearClassifier(D1,w_D1);
%%
% Number of elements corretly classified with D1:
D1_correct_class = sum(y_classified_D1 == diabetes_db.y)
correct_rate = (D1_correct_class/length(diabetes_db.y))*100

%%
[ w_D2, costFunction_D2 ] = gradient_descent( D2, diabetes_db.y, maxIter, minError, w_in, t);
[ y_classified_D2 ] = linearClassifier(D2,w_D2);

%%
% Number of elements corretly classified with D2:
D2_correct_class = sum(y_classified_D2 == diabetes_db.y)
correct_rate = (D2_correct_class/length(diabetes_db.y))*100

%% Normal Vectors
%Normal vector of the hyperplane for Dataset1
w_D1
%Normal vector of the hyperplane for Dataset1
w_D2

%% 
% * 1) Repeat the learning process in block 3 using just D2 but holding-out the last
%fifth of the data set for testing purposes, i.e. use the first 4/5-th for train and
%the last 1/5-th for testing.

%%
% a) Clear workspace
clear all;
close all;
clc;

%% 
% b)Preprocess data
diabetes_db = load('diabetes.mat');
D2 = preprocessData2(diabetes_db.x,diabetes_db.y);

%%
% c) Split data in two sets 
train_size = round(size(D2,2)*4/5);
train_data = D2(:,1:train_size);
train_target = diabetes_db.y(1:train_size);
test_data = D2(:,train_size+1:end);
test_target = diabetes_db.y(train_size+1:end);
%% 
% d) Train model with the training set
w_in = ones(1,size(D2,1)+1);
maxIter = 200000;
minError = 1e-5;
t = 0.000001;
[ w_train, costFunction_train ] = gradient_descent( train_data, train_target, maxIter, minError,w_in,t );
[ y_classified_train ] = linearClassifier(train_data,w_train);
[ y_classified_test ] = linearClassifier(test_data,w_train);


%%
% e) Answer the following questions:   
%   - Which is the error rate on your training data?
train_errors = sum(y_classified_train ~= train_target);
train_err_rate = (train_errors/train_size) *100

%   - Which is the error rate on your test data?
test_errors = sum(y_classified_test ~= test_target);
test_err_rate = (test_errors/length(test_target)) *100

%   - Are they similar? Did you expect that behavior? Why?
% Yes, they are similar
%THIS EXPLANATION IS TEMPORAL JUST WRITING BASIC IDEA
% Yes that behaviour was expected because we have replaced the Nan values
% with the mean value of the attribute, for this the examples we reserve
% for the test will be "similar" to those that we have used in the
% training, achieving a good classification


%% 
%  * 1) Repeat the process in block 4 changing the order of some of the
% steps. Follow exactly the following steps in your process:

%% 
% a) Clear workspace
clear all;
close all;
clc;

%% 
% b) Split your data in two sets: training and testing
data = load('diabetes.mat');
train_size = round(size(data.x,2)*4/5);
train_data = data.x(:,1:train_size);
train_target = data.y(1:train_size);
test_data = data.x(:,train_size+1:end);
test_target = data.y(train_size+1:end);

%%

% c) Preprocess the data replacing the NaN using the method for creating
% D2. But this time use only the data corresponding to the training set.
D2_train = train_data;

for i=1:size(train_data,1)
    attr = train_data(i,:);
    attr_no_nan = ~isnan(train_data(i,:));
    % c is a vector containing 1 or -1 in those positions that correspond
    % to the ith instance of data, describing its belonging to class 1 or
    % -1. On the other hand, a 0 corresponds to a NaN element.
    c = train_target .* attr_no_nan';
    c_nan = train_target .* ~attr_no_nan';

    c1_not_nan(i) = sum(attr(find(c == 1))) / length(find(c==1));
    c2_not_nan(i) = sum(attr(find(c == -1))) / length(find(c==-1));
    
    D2_train(i,c_nan == 1) = c1_not_nan(i);
    D2_train(i,c_nan == -1) = c2_not_nan(i);
end

%%
% d) Train your model on the training set.
w_in = ones(1,size(D2_train,1)+1);
maxIter = 200000;
minError = 1e-5;
t = 0.000001;
[ w_train, costFunction_train ] = gradient_descent( D2_train, train_target, maxIter, minError,w_in,t );
[ y_classified_train ] = linearClassifier(train_data,w_train);

%%
% e) Replace the NaN values using the means computed on the training data
D2_test = test_data;

for i=1:size(test_data,1)
    attr = test_data(i,:);
    attr_no_nan = ~isnan(test_data(i,:));
    c = test_target .* attr_no_nan';
    c_nan = test_target .* ~attr_no_nan';
    
    D2_test(i,c_nan == 1) = c1_not_nan(i);
    D2_test(i,c_nan == -1) = c2_not_nan(i);
end

%%
% f) Answer the following questions: 
%   - Which is the error rate on your training data? 
train_errors = sum(y_classified_train ~= train_target);
train_err_rate = (train_errors/train_size) *100

%   - Which is the error rate on your test data? 
[ y_classified_test ] = linearClassifier(D2_test,w_train);
test_errors = sum(y_classified_test ~= test_target);
test_err_rate = (test_errors/length(test_target)) *100

%   - Are they similar? Did you expect that behavior? Why?

%%
% g) Compare these results with the ones in block 4. Do we achieve
%   better or worse results? Why?
%BETTER RESULTS