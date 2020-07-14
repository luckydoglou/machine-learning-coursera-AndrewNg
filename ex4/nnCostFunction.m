function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% compute hypothesis h(x)
% input layer a1
X = [ones(m, 1) X];   % 5000*401
% hidden layer a2 is 5000x25
z2 = X * Theta1';     % (5000*401)*(401*25)=5000*25
a2 = sigmoid(z2);      % 5000*25
a2 = [ones(m, 1) a2];  % 5000*26

% output layer a3 is 5000x10
z3 = a2 * Theta2';     % (5000*26)*(26*10)=5000*10
a3 = sigmoid(z3);      % 5000*10
h_theta = a3;

% since h_theta has 10 columns, which means it's 10 h_theta(i) combined
% so we have to make sure y matches the h_theta's dimensions
% y need to be 5000*10 containning 0s and 1s.
new_y = zeros(m, num_labels); % m is 5000, num_labels is 10

% now we need to map the results from y that passed in as parameter to new_y
% make sure 1s are placed in correct place and rest places are 0s
for i = 1:m
    new_y(i, y(i)) = 1;  
end

% finally, do element-wise calculate for cost function, we get J is 5000*10
% two sum(). First sum adds each row from one column together (1*10), second 
% sum adds each column together (1*1)
% Note: the reason why we do element-wise calculation is because we want to
% sum up the relavent values for each digit that relavent to where y(i) 
% equals 1. 
J = 1/m * sum( sum( (-new_y .* log(h_theta) - (1 - new_y) .* log(1 - h_theta)) ) );

% Theta1 25*401, Theta2 10*26
% remove the bias term, the first column
t1 = Theta1(:, 2:size(Theta1, 2));   % 25*400
t2 = Theta2(:, 2:size(Theta2, 2));   % 10*25
% calculate the regularization
reg = lambda/(2*m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));
% update cost function
J = J + reg;


% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% one training example at a time for step 1 through step 4
for t = 1:m
    % step 1, compute a1, z2, a2, z3, a3 at t
    %X = [ones(m, 1) X]; % 5000*401
    a1 = X(t, :);  % 1*401
    z2 = a1 * Theta1';  % (1*401)*(401*25)=1*25
    a2 = sigmoid(z2);  % 1*25
    
    a2 = [1, a2];  % 1*26, add bias term 
    z3 = a2 * Theta2';  % (1*26)*(26*10)=1*10
    a3 = sigmoid(z3);  % (1*10)
    
    % step 2, calculate and set delta of layer 3 (output layer)
    % new_y (5000*10) is declared in part 1
    delta_3 = a3 - new_y(t, :)  % (1*10)-(1*10)=1*10
    
    % step 3, for hidden layer l = 2, calculate and set delta of layer 2,
    % the hidden layer
    z2 = [1, z2];  % 1*26, add bias term
    delta_2 = (delta_3 * Theta2) .* sigmoidGradient(z2);  % (1*10)*(10*26).*(1*26)=1*26
    
    % step 4, accumulate the gradient
    delta_2 = delta_2(2: end);  % 1*25, remove bias term
    

    Theta2_grad = Theta2_grad + delta_3' * a2;  % (10*26)+(10*1)*(1*26)=10*26
    Theta1_grad = Theta1_grad + delta_2' * a1;  % (25*401)+(25*1)*(1*401)=25*401
    
end

% step 5, obtain the unregularized gradient for cost function
Theta2_grad = (1/m) * Theta2_grad;  % 10*26
Theta1_grad = (1/m) * Theta1_grad;  % 25*401


% -------------------------------------------------------------
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

reg2 = (lambda/m) * Theta2;
reg1 = (lambda/m) * Theta1;

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + reg2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + reg1(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
