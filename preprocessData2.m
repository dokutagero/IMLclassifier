function [ D2 ] = preprocessData2( data, target )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
D2 = data;
for i=1:size(data,1)
    attr = data(i,:);
    attr_no_nan = ~isnan(data(i,:));
    % c is a vector containing 1 or -1 in those positions that correspond
    % to the ith instance of data, describing its belonging to class 1 or
    % -1. On the other hand, a 0 corresponds to a NaN element.
    c = target .* attr_no_nan';
    c_nan = target .* ~attr_no_nan';

    c1_not_nan(i) = sum(attr(find(c == 1))) / length(find(c==1));
    c2_not_nan(i) = sum(attr(find(c == -1))) / length(find(c==-1));
    
    D2(i,c_nan == 1) = c1_not_nan(i);
    D2(i,c_nan == -1) = c2_not_nan(i);
end
end

