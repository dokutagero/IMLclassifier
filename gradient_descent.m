function [ output_args ] = gradient_descent( x, y, maxIter, minError, w)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Slide "31", not clear why there is a 2 in the denominator of the cost
%funciton
iter = 0;
N = size(x,2);

while iter < maxIter
    x = [ones(1,size(x,2));x];
    grad = 1/N * (x'*w-y)
    J = 1/(2*N) * (x'*w-y)'(x'*w-y); %Cost function
    w = w - t*grad*

end

end

