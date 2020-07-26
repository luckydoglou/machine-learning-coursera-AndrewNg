function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

n = size(X, 1);

for i = 1:n
    
    % set min equal to euclidian distance of current points to 1st centroid
    min = sum((centroids(1, :) - X(i, :)) .^ 2);
    pos = 1; % index of current closest centroid
    
    for j = 1:K
        
        % update min and closest centroid index
        if sum((centroids(j, :) - X(i, :)) .^2) < min
            min = sum((centroids(j, :) - X(i, :)) .^2);
            pos = j;
        end
    
    idx(i) = pos;
    end
    
end


% =============================================================

end

