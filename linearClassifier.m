function [ y ] = linearClassifier( x,w )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

X = [ones(1,size(x,2));x];
y = sign(X'*w);


end

