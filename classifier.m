%% EXERCISE 2: CLASSIFIER
% * Submitted by: Abhishek Saurabh, Juanjo Rubio and Miguel M?ndez
% * Program: Master's in Artificial Intelligence
% * Course: Introduction to Machine Learning
% * Due date: 29-Nov-2015

%% Data set analysis
diabetes_db = load('diabetes.mat');
%%
% * Which is the cardinality of the training set?
%
cardinality = size(diabetes_db.x,2)
%%
% * Which is the dimensionality of the training set?
%
dimensionality = size(diabetes_db.x,1)
%%
% * Which is the mean value of the training set?
%mean_diabetes = mean2(diabetes_db.x)
mean_diabetes_attr = mean(diabetes_db.x,2)
%%
% * 1) Create a new dataset D1, replacing the NaN values with the mean
% value of the corresponding attribute without considering the missing values.
d1 = diabetes_db.x;
for i=1:size(diabetes_db.x,2)
    nan_pos = isnan(diabetes_db.x(:,i));
    nan_pos_inv = ~nan_pos;
    %attr_mean(i) = mean(diabetes_db.x(nan_pos,i));
    d1(nan_pos,i) = mean(diabetes_db.x(nan_pos_inv,i));
end


