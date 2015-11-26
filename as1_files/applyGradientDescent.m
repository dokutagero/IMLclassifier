function [ cellTheta, iterCount, error, objFuncHistory] = ...
        applyGradientDescent( X, y, theta, learningRate, numberIterations, minImprovement )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
m = size(X, 1);
cellTheta = cell(numberIterations, 1);
cellTheta{1} = theta; % To hold values of theta for plotting objective function vs parameters.
objFuncHistory = zeros(numberIterations, 1); % To hold objective function at each iteration
for i = 2:numberIterations
    error = y- X*cellTheta{i-1};
    gradientCostFunction = -(2/m)* X' * error;
    cellTheta{i} = cellTheta{i-1} - learningRate * gradientCostFunction;
    objFuncHistory(i) = getObjectiveFunction (X, y, cellTheta{i});
    if i>1
        if abs(objFuncHistory(i)- objFuncHistory(i-1))< minImprovement
            break
        end
    end
end
iterCount = i;
end
