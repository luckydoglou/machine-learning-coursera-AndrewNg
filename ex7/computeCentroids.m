function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m, n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% iterate through every centroid
for i = 1:K
    
    % initilize new centroid
    centroid_new = zeros(1, n);
    count = 0;
    % find every example that has been assigned to this centroid
    for j = 1:m
         
        if idx(j) == i
            centroid_new = centroid_new + X(j, :);
            count = count + 1;  
        end  
    end
    
    % update centroid position
    centroids(i, :) = centroid_new ./ count;
    
end

% ================
% alternative way
% for i = 1:K
%     centroids(i, :) = mean(X(idx==i, :));

% =============================================================


end

