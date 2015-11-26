function [ objectiveFunction ] = ...
                    getObjectiveFunction( matIndependentVar, matDependentVar, theta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% The mean is halved (1/2m) as a convenience for the computation of the
% gradient descent, as the derivative term of the square function will
% cancel out the 1/2 term.
m = size(matDependentVar, 1);
objectiveFunction = (1/(2*m))*sum((matDependentVar -  matIndependentVar * theta).^2);
end

