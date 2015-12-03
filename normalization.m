function [meanColumns, stdevColumns, normalizedData] = normalization(matData)
% Description: Function takes data contained in a matrix and returns
% normalized data. Normalization is done by calculating the z-score. 
% While normalizing the data in this manner it is assumed that the data 
% is normally distributed. If the standard deviation of a column is zero,
% meaning thereby that all the elements are the same, function would return
% a zero in such a case.
% Input argument : A matrix containing data without any missing values.
% Output arguments: (1) Mean of each column of matrix as Double.
%                   (2) Standard Deviation of each column of matrix as
%                   Double.
%                   (3) Normalized data in a Matrix.
% Example: 
% [columnMeans, columnStdev, normalizedData] = normalization(matrixData)
normalizedData = [];
sizeMatData = size(matData); 
meanColumns = mean(matData);
stdevColumns = std(matData);
for nCol = 1:sizeMatData(2)
    if stdevColumns(:,nCol) == 0
        normalizedData(:,nCol) = 0;
    else
        normalizedData(:,nCol) = (matData(:,nCol) - meanColumns(nCol))/stdevColumns(nCol);
    end 
end