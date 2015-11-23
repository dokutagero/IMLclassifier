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
%%
% * 2) Create a new dataset D2, replacing the NaN values with the mean 
% value of the corresponding attribute without considering the missing 
% values conditioned to the class they belong, i.e. replace the missing 
% attribute values of class +1 with the mean of that attribute of the 
% examples of class +1, and the same for the other class.

d2 = diabetes_db.x;

% meanc1 contains the mean of each column without considering the NaN value
% for instances belonging to class +1.
% meanc2 contains the mean of each column without considering the NaN value
% for instances belonging to class -1;
meanc1 = zeros(size(diabetes_db.x,1),1);
meanc2 = zeros(size(diabetes_db.x,1),1);
for i=1:size(diabetes_db.x,1)
    idx_c1=0;
    idx_c2=0;
    for j=1:size(diabetes_db.x,2)
        if ~isnan(diabetes_db.x(i,j))
            if diabetes_db.y(i) == 1
                meanc1(i) = meanc1(i) + diabetes_db.x(i,j);
                idx_c1 = idx_c1 + 1;
            else
                meanc2(i) = meanc2(i) + diabetes_db.x(i,j);
                idx_c2 = idx_c2 + 1;
            end
        end
    end
    if idx_c1 ~= 0 
        meanc1(i) = meanc1(i) / idx_c1;
    end
    if idx_c2 ~= 0 
        meanc2(i) = meanc2(i) / idx_c2;
    end
end

%meanc1_total = mean(meanc1);
%meanc2_total = mean(meanc2);

% Substitute each NaN value with the corresponding mean for each class.
% for i=1:size(diabetes_db.x,2)
%     nan_pos = isnan(diabetes_db.x(:,i));
%     nan_pos_inv = ~nan_pos;
%     if diabetes_db.y(i)==1
%         d2(nan_pos,i) = meanc1_total;
%     else
%         d2(nan_pos,i) = meanc2_total;
%     end
% end

