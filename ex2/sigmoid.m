function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% For large positive values of x, the sigmoid should be close to 1, 
% while for large negative values, the sigmoid should be close to 0. 
% Evaluating sigmoid(0) should give you exactly 0.5
g = 1 ./ (1 + exp(-z));



% =============================================================

end
