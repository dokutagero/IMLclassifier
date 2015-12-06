%% EXERCISE 2: OUR FIRST CLASSIFIER
% * Submitted by: Abhishek Saurabh, Juanjo Rubio and Miguel M?ndez
% * Program: Master's in Artificial Intelligence
% * Course: Introduction to Machine Learning
% * Due date: 06-Dec-2015

%% B. Data set analysis
%%
diabetes_db = load('diabetes.mat');
%%
% |*Question block 1:*|
%%
% |*1) Which is the cardinality of the train set?*|

% Cardinality of the training dataset
cardinality = size(diabetes_db.x,2)
%%
% |*2) Which is the dimensionality of the train set?*|

% Dimensionality of training dataset
dimensionality = size(diabetes_db.x,1)
%%
% |*3) Which is the mean value of the train set?*|

% Mean value of training dataset
mean_diabetes_attr = mean(diabetes_db.x,2)
%%
% |*Question block 2:*|
%%
% |*1) Create a new dataset D1, replacing the NaN values with the mean
% value of the corresponding attribute without considering the missing
% values.*|

D1 = diabetes_db.x;
for i=1:size(diabetes_db.x,1)
    attr = diabetes_db.x(i,:);
    nan_pos = isnan(diabetes_db.x(i,:));
    attr_no_nan = ~nan_pos;
    D1(i,nan_pos) = mean(diabetes_db.x(i,attr_no_nan));
end
%%
% |*2) Create a new dataset D2, replacing the NaN values with the mean value
% of the corresponding attribute without considering the missing values
% conditioned to the class they belong, i.e. replace the missing attribute
% values of class +1 with the mean of that attribute of the examples of
% class +1, and the same for the other class.*|

D2 = preprocessData(diabetes_db.x,diabetes_db.y);
%%
% |*3) [Optional :] Explain another method to deal with missing values and
% apply it to preprocess the training data. Include the reference of the
% method used. Consider this new dataset as D3.*|
 
%%
% |KNN Imputation: In this method similarity between two instances of data
% is measured to impute the missing values. It is assumed that if two
% instance of data is similar and one of them has a missing value in some
% variable, then the probability that this value is similar to the value of
% some other observation is high.| 
%%
% |*Reference:* Batista, Gustavo EAPA, and Maria Carolina Monard. "A Study of
% K-Nearest Neighbour as an Imputation Method." HIS 87.251-260 (2002): 48.|
%%
% |knnimpute() function in matlab implements this method. This function
% replaces NaNs in the data with a weighted mean of k nearest neighbours
% (=5 in this case). Weights are inversely proportional to the distance
% from the neighbouring columns.|
D3 = knnimpute(diabetes_db.x', 5)';
%%
% |*4) Which are the new mean values of each dataset?*|

mean_d1 = mean(D1,2)
mean_d2 = mean(D2,2)
mean_d3 = mean(D3,2)

%% C. A Simple Classifier
%%
% |*Question block 3:*|
%%
% |*1) In this model you have to learn the threshold value. Explain how you
% can accomodate this parameter.*|

%Normalize data 
[meanColumns, stdevColumns, D1_norm] = normalization(D1');
D1_norm = D1_norm';
[meanColumns, stdevColumns, D2_norm] = normalization(D2');
D2_norm = D2_norm';
[meanColumns, stdevColumns, D3_norm] = normalization(D3');
D3_norm = D3_norm';
%Set the initial parameters
w_in = zeros(1,size(D1,1)+1)*10;
maxIter = 10000;
minError = 1e-5;
t = 0.01;

%Obtain the linear regression using gradient descend in all the datasets.
[ w_D1, costFunction_D1 ] = ...
    gradient_descent( D1_norm, diabetes_db.y, maxIter, minError,w_in,t );
[ y_classified_D1 ] = linearClassifier(D1_norm,w_D1);


[ w_D2, costFunction_D2 ] = ...
    gradient_descent( D2_norm, diabetes_db.y, maxIter, minError, w_in, t);
[ y_classified_D2 ] = linearClassifier(D2_norm,w_D2);


[ w_D3, costFunction_D3 ] = ...
    gradient_descent( D3_norm, diabetes_db.y, maxIter, minError, w_in, t);
[ y_classified_D3 ] = linearClassifier(D3_norm,w_D3);
%%
% Considering that our classifier uses a linear regression method for
% classifying our data, we define the belonging to the class 1 like the
% following expression
%%
% $$ \sum_i^dw_{i}x_{i}>threshold $$
%%
% While the belonging to the -1 class is defined as
%%
% $$ \sum_i^dw_{i}x_{i}\leq threshold $$
%%
% Then, we define our linear classifier function as follows:
%%
% $$ h(x) = sign(\sum_i^dw_{i}x_{i}+w_{0}) $$
%%
% Where $$ w_{0} $$ is  $$ -threshold $$.
%%
%%
D1_threshold = -w_D1(1)
D2_threshold = -w_D2(1)
D3_threshold = -w_D3(1)

%%
% |*2) Report the normal vector of the separating hyperplane for each
% data set D1, D2, D3.*|
%%
%Normal vector of the hyperplane for Dataset1
w_D1
%Normal vector of the hyperplane for Dataset2
w_D2
%Normal vector of the hyperplane for Dataset3
w_D3
%%
% |*3) Compute the error rates achieved on the training data. Are there
% significant differences?*|

%%
D1_correct_class = sum(y_classified_D1 == diabetes_db.y);
correct_rate_D1 = (D1_correct_class/length(diabetes_db.y))*100;
error_rate_D1 = 100 - correct_rate_D1

% Number of elements corretly classified with D2:
D2_correct_class = sum(y_classified_D2 == diabetes_db.y);
correct_rate_D2 = (D2_correct_class/length(diabetes_db.y))*100;
error_rate_D2 = 100 - correct_rate_D2


% Number of elements corretly classified with D2:
D3_correct_class = sum(y_classified_D3 == diabetes_db.y);
correct_rate_D3 = (D3_correct_class/length(diabetes_db.y))*100;
error_rate_D3 = 100 - correct_rate_D3

%%
% The error rates where calculated using the labels provided with the
% dataset. Comparing one by one the elements classified in the training
% phase with the actual labels provided. There are no significant
% differences between the datasets, being D2 the one with better results.
%%
% |*Question block 4*|
%%
% |*1) Repeat the learning process in block 3 using just D2 but holding-out
% the last fifth of the data set for testing purposes, i.e. use the first
% 4/5-th for train and the last 1/5-th for testing.*|

%%
% a) Clear workspace
clear all;
close all;
clc;

%% 
% b)Preprocess data replacing NaN using the method for creating D2.
diabetes_db = load('diabetes.mat');
D2 = preprocessData2(diabetes_db.x,diabetes_db.y);

%%
% c) Split data in two sets: first 4/5-th is to be used for training and
% the last 1/5-th will be used for testing purposes.
[meanColumns, stdevColumns, D2] = normalization(D2');
D2 = D2';

train_size = round(size(D2,2)*4/5);
train_data = D2(:,1:train_size);
train_target = diabetes_db.y(1:train_size);
test_data = D2(:,train_size+1:end);
test_target = diabetes_db.y(train_size+1:end);
%% 
% d) Train model on the training set
w_in = zeros(1,size(D2,1)+1);
maxIter = 10000;
minError = 1e-5;
t = 0.01;
[ w_train, costFunction_train ] = gradient_descent( train_data, train_target, maxIter, minError,w_in,t );
% Classification of training and test with the weights obtained using the
% training dataset.
[ y_classified_train ] = linearClassifier(train_data,w_train);
[ y_classified_test ] = linearClassifier(test_data,w_train);




%%
% e) Answer the following questions:   
%   - Which is the error rate on your training data?
train_errors = sum(y_classified_train ~= train_target);
train_err_rate = (train_errors/train_size) *100;

%   - Which is the error rate on your test data?
test_errors = sum(y_classified_test ~= test_target);
test_err_rate = (test_errors/length(test_target)) *100;


train_err_rate
test_err_rate
%%
% The values obtained for the training error rate and test error rate are
% available above. Both error rates are very similar, having only a
% difference of around 1% error rate. This behaviour was expected since
% when replacing the NaN values we are introducing the mean with values
% from the training dataset taken into account. We are averaging the values
% for a given attribute with both the values from the training and the
% testing dataset, making them more similar.
%%
% |*Question block 5*|
%%

% |*1) Repeat the process in block 4 changing the order of some of the
% steps. Follow exactly the following steps in your process:*|

%% 
% a) Clear workspace
clear all;
close all;
clc;

%% 
% b) Split data in two sets: first 4/5-th is to be used for training and
% the last 1/5-th will be used for testing purposes.
data = load('diabetes.mat');
train_size = round(size(data.x,2)*4/5);
train_data = data.x(:,1:train_size);
train_target = data.y(1:train_size);
test_data = data.x(:,train_size+1:end);
test_target = data.y(train_size+1:end);

%%

% c) Preprocess the data replacing the NaN using the method for creating
% D2. But this time use only the data corresponding to the training set.

%c1_means have values of the dataset that belong to class -1 while c2_means
%has the value for class -1
[D2_train,c1_means,c2_means] = preprocessData2(train_data,train_target);

%In this case, beside normalizing the training dataset, we obtain the mean
%and standard deviation from it for normalizing the test dataset.
[meanColumns, stdevColumns, D2_train] = normalization(D2_train');
D2_train = D2_train';

%%
% d) Train your model on the training set.
w_in = zeros(1,size(D2_train,1)+1);
maxIter = 10000;
minError = 1e-5;
t = 0.01;
[ w_train, costFunction_train ] = gradient_descent( D2_train, train_target, maxIter, minError,w_in,t );
[ y_classified_train ] = linearClassifier(D2_train,w_train);

%%
% e) Replace the NaN values using the means computed on the training data
D2_test = test_data;

%Note that c1_means and c2_means was computed above and correspond to the
%mean of the attributes taking into account the class each sample belongs
%to.
for i=1:size(test_data,1)
    attr = test_data(i,:);
    attr_no_nan = ~isnan(test_data(i,:));
    c = test_target .* attr_no_nan';
    c_nan = test_target .* ~attr_no_nan';
    
    D2_test(i,c_nan == 1) = c1_means(i);
    D2_test(i,c_nan == -1) = c2_means(i);
end

%Normalize the test dataset with the mean and standard deviation from the
%training dataset.
[meanColumns, stdevColumns, D2_test] = normalization(D2_test',meanColumns, stdevColumns);
D2_test = D2_test';

%%
% f) Answer the following questions: 
%   - Which is the error rate on your training data? 
train_errors = sum(y_classified_train ~= train_target);
train_err_rate = (train_errors/train_size) *100;

%   - Which is the error rate on your test data? 
[ y_classified_test ] = linearClassifier(D2_test,w_train);
test_errors = sum(y_classified_test ~= test_target);
test_err_rate = (test_errors/length(test_target)) *100;
%%
train_err_rate
test_err_rate

%   - Are they similar? Did you expect that behavior? Why?

%%
% g) Compare these results with the ones in block 4. Do we achieve
%   better or worse results? Why?
%BETTER RESULTS
%%
% |*Question block 6*|
%%

% |*1) Repeat the process in block 5 changing the percentage of the data for
% training and testing. Plot a graph with the training and test error rates
% for each splitting percentage point. Comment the results.*|
clear all
times = 10;

for i = 1:times-1
   block5;    
   train_rate(i) = train_err_rate/100;
   test_rate(i) = test_err_rate/100;
end

figure;
plot((1:times-1),train_rate);
hold on;
plot((1:times-1),test_rate,'r');
legend('Train error rate','Test error rate');

%% Testing analytical solution
clear all;
times=10;
%size of training
N = zeros(1,times-1);

for i = 1:times-1
   block5_analitic;
   train_rate(i) = train_err_rate/100;
   test_rate(i) = test_err_rate/100;
end

figure;
plot((1:times-1),train_rate);
hold on;
plot((1:times-1),test_rate,'r');
legend('Train error rate','Test error rate')

%%
% 2)

%Not sure if log10, log (in matlab is ln) or log2
upperBound = zeros(1,length(N));
dVC = 9;
delta = 0.01;
for j=1:length(N);
    upperBound(j) =  train_rate(j) + sqrt((dVC*(log(2*N(j)/dVC)+1) + log(2/delta))/(2*N(j)));
end
hold on
plot(upperBound,'g');

%%
% 3)
syms n n2;
delta = 0.05;
error_deviation = 0.01;
n = vpasolve(sqrt((dVC*(log(2*n/dVC)+1) + log(2/delta)/(2*n))) == ...
    error_deviation, n);
%n = solve(error_deviation == sqrt((dVC*(log(2*n/dVC)+1) + log(2/delta)/(2*n))), n);
n2 = solve(error_deviation == sqrt((log((2*n2*exp(1) / dVC)^dVC * (2/delta)))/(2*n2)), n2) ;

eval(n)
eval(n2)

% clear; clc;
% delta = 0.05; % confidence 95%
% deviationError = 0.01; % variance 1%
% dVC = 9;
% 
% syms N;
% sol1 = solve(sqrt( log( ((((2*N*exp(1))/dVC)^dVC) * (2/delta) ) - exp(2*N) )) == deviationError, N) ;
% fprintf('Result 1: \n');
% disp(sol1);
%Some notes: slide 61/73 has the expression of VC. From the data given in
%the last question, the % error deviation should be the difference between
%the training and testing. On the other hand, the confidence is 1-delta. I
%supposed that isolation the n value, which is the number of training
%samples that would give the given error dev and confidence. After some
%discussion with Marco, it seems that the question is not actually this
%one. We are not sure exactly how to proceed.

