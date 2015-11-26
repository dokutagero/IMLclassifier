function [ d,y_out ] = linearClassifierPreprocessor( x,y )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

d = zeros(size(x));
y_out = zeros(size(y));

class_1_ind = find(y == 1);
class_m1_ind = find(y == -1);
y_out(1:length(class_m1_ind)) = y(class_m1_ind);
d(:,1:length(class_m1_ind)) = x(:,class_m1_ind);
y_out((length(class_m1_ind)+1):end) = y(class_1_ind);
d(:,(length(class_m1_ind)+1):end) = x(:,class_1_ind);


end

