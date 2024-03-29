function [ w_out, costFunction ] = gradient_descent( x, y, maxIter, minError, w, t)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Slide "31", not clear why there is a 2 in the denominator of the cost
%funciton
w = w(:);
y = y(:);
iter = 1;
N = size(x,2);
X = [ones(1,N);x];
costFunction = zeros(maxIter,1);


while iter<=maxIter
    % errors = f(x;w)-y
    errors = y-X'*w;
    % CostFunction = 1/2N*sum(f(x;w)-y)^2). We are trying to minimize this
    % function that represents the error of the predicted vs the labeled.
    costFunction(iter) = getCostFunction(X,w,y);
    % grad is the gradient of the cost function. The minus sign represents
    % the descending direction of the gradient.
    %grad = (1/N)*errors*X;
    grad = -(1/N)*X*errors;
    % t describes the step size when descending to the minimum following
    % the gradient.
    w = w - t*grad;
    iter = iter + 1;

end

w_out = w;

end

