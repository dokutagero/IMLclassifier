function [ D, c1_not_nan, c2_not_nan ] = preprocessData( data, target )
% Description: Function takes a given dataset and processes the NaN values
% present in it. It finds the position of the NaN values and substitutes
% them with the mean of the given attribute taking into account the class
% that the given sample of data belongs to.
% Input argument : dataset and labels.
% Output arguments: (1) Dataset processed without NaN values.
%                   (2) Mean of the values of an attribute for class +1
%                   (3) Mean of the values of an attribute for class -1
% Example: 
% 
D = data;
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
    
    D(i,c_nan == 1) = c1_not_nan(i);
    D(i,c_nan == -1) = c2_not_nan(i);
end
end
