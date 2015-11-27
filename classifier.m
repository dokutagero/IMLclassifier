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

D2 = diabetes_db.x;

for i=1:size(diabetes_db.x,1)
    attr = diabetes_db.x(i,:);
    attr_no_nan = ~isnan(diabetes_db.x(i,:));
    % c is a vector containing 1 or -1 in those positions that correspond
    % to the ith instance of data, describing its belonging to class 1 or
    % -1. On the other hand, a 0 corresponds to a NaN element.
    c = diabetes_db.y .* attr_no_nan';
    c_nan = diabetes_db.y .* ~attr_no_nan';

    c1_not_nan(i) = sum(attr(find(c == 1))) / length(find(c==1));
    c2_not_nan(i) = sum(attr(find(c == -1))) / length(find(c==-1));
    
    D2(i,c_nan == 1) = c1_not_nan(i);
    D2(i,c_nan == -1) = c2_not_nan(i);
end

%%
% * 4) Which are the new mean values of each dataset?

mean_d1 = mean(D1,2)
mean_d2 = mean(D2,2)

%% A simple classifier
%
%%
% * 1) In this model you have to learn the threshold value. Explain how you
% can accomodate this parameter.
[ w_D1, costFunction_D1 ] = gradient_descent( D1, diabetes_db.y, 200000, 1e-5, [1,1,1,1,1,1,1,1,1], 0.000001);
[ y_classified_D1 ] = linearClassifier(D1,w_D1);
%%
% Number of elements corretly classified with D1:
D1_correct_class = sum(y_classified_D1 == diabetes_db.y)
correct_rate = (D1_correct_class/length(diabetes_db.y))*100
%%
[ w_D2, costFunction_D2 ] = gradient_descent( D2, diabetes_db.y, 200000, 1e-5, [1,1,1,1,1,1,1,1,1], 0.000001);
[ y_classified_D2 ] = linearClassifier(D2,w_D2);
%%
% Number of elements corretly classified with D1:
D2_correct_class = sum(y_classified_D2 == diabetes_db.y)
correct_rate = (D2_correct_class/length(diabetes_db.y))*100
