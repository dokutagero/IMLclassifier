function [ costFunction ] = getCostFunction( X,w,y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(X,2);
costFunction = (1/(2*N))*(X'*w-y)'*(X'*w-y);

end

