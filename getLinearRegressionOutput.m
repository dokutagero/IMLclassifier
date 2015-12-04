function [ weights, y_predicted ] = getLinearRegressionOutput( x, y )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% Linear regression with intercept
x_modified = [ones(1,size(x,2));x];
weights = (x_modified*x_modified') \ (x_modified*y);

%weights = pinv(x_modified)*y';
y_predicted = x_modified' * weights;
end


