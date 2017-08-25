
function [J grad] = nnCostFunction(nn_params, ...
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
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1trans = Theta1';

Theta2trans = Theta2';
%%FWD prop
X = [ones(m,1), X];
z2 = X*Theta1trans;
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*Theta2trans;
a3 = sigmoid(z3);
h = a3';

%%convert output
% Y = [ismembc(y,1) ismembc(y,2) ismembc(y,3)...
%     ismembc(y,4) ismembc(y,5) ismembc(y,6)...
%     ismembc(y,7) ismembc(y,8) ismembc(y,9)...    Faster
%     ismembc(y,10)];
Y = zeros(num_labels, m);
for i = 1:num_labels
    Y(i,:) = ismembc(y,i);
end 


%%Cost
J = sum(sum((log(h).*(-Y)-log(1-h).*(1-Y))./m));

%%Reg Cost
Theta1sq = Theta1(:,2:end).^2;
Theta2sq = Theta2(:,2:end).^2;

J = J + (sum(Theta1sq(:)) + sum(Theta2sq(:))).*(lambda/(2*m));

%%Back Prop
d3 = h - Y;
d2 = (Theta2trans(2:end,:)*d3).*sigmoidGradient(z2)';
D1 = d2*X;
D2 = d3*a2;

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad = (1/m)*D1+(lambda/m)*Theta1;
Theta2_grad = (1/m)*D2+(lambda/m)*Theta2;


grad = [Theta1_grad(:); Theta2_grad(:)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
