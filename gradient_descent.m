function [ output_args ] = gradient_descent( x, y, maxIter, minError, w, t)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Slide "31", not clear why there is a 2 in the denominator of the cost
%funciton
iter = 0;
N = size(x,2);

while iter < maxIter
    % the ones added to x will take care of the independent term w0. 
    x = [ones(1,size(x,2));x];
    % x is N * (dimensions + 1)
    % y is N * 1
    grad = 1/N * x * (x'*w-y);
    % grad is (dim+1) * 1
    % w is (dim+1) * 1
    %J = 1/(2*N) * (x'*w-y)'(x'*w-y); %Cost function
    w = w - t*(-grad);
    iter = iter + 1;

end

end

