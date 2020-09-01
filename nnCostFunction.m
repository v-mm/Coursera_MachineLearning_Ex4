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

% NOTE:
% Calculations in linear regression / logistic regression for linear hypothesis
% use vector computations - where the training examples, parameters (small case 
% theta)and outputs are vector columns (or handled as vectors - x(i) being 
% a row vector in X).
% However for non-linear hypothesis using neural networks, the calculations involve
% matrix computaions - where the input layer, hidden layers, output layers and
% parameters (capital Theta) and gradients are all matrices. These matrices are
% either unrolled as vectors or reshaped back from vectors into matrices as the
% case requires.


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



% ====================================================
% my code here
% ====================================================

% cost J(?) = 
% -(1/m) * SumOver_i_1_m of SumOver_k_1_K [ (y_i_k) * log(hThetaX_i_k) + ...
%  (1-y_i_k) * log(1-hThetaX_i_k) ]

% where y, hThetaX are matrices, not vectors
% and y = m x k matrix (m examples,k classes)
% and hThetaX = m x k matrix

% so for the double summation (SumOver_i_1_m of SumOver_k_1_K) 
% across all elements of the matrix, do element wise multiplication, then sum 
% the result elements for a scalar cost term.

% from the data file, y is a vector 5000 x 1 with 10 labels / classes
% to compute cost for all y(i) examples, we convert y to a matrix such that
% y_matrix is 5000 x 10 (i.e. m x num_labels) where for each row of y_matrix,
% there is a 1 at the index corresponding to a label in the y vector
% e.g. if y = [2,3,4,2,3,2] with num_labels = 4
% y_matrix = 
% [0 1 0 0] % 1 in 2nd column, 2 is 1st element of y vector
% [0 0 1 0] % 1 in 3rd column, 3 is next element of y vector
% [0 0 0 1] % 1 in 4th column, 4 is next element of y vector
% [0 1 0 0] % 1 in 2nd column, 2 is next element of y vector
% [0 0 1 0] % 1 in 3rd column, 3 is next element of y vector
% [0 1 0 0] % 1 in 2nd column, 2 is next element of y vector
I = eye(num_labels);
y_matrix = I(y,:);
% y_matrix is (m x num_labels) i.e. 5000 x 10
% each row y(i=1 to 5000) indicates one among 10 labels (e.g '10','7',..'2','1'. 
% Thus 5000 examples of 10 different labels

% some variables
HThetaXik = zeros(m, num_labels);
One_HThetaXik = zeros(m, num_labels);
YikLogHThetaXik = zeros(m, num_labels);
One_YikLog1_HThetaXik = zeros(m, num_labels);

% setting bias Thetas to 0
Theta1_BiasZero = Theta1;
Theta2_BiasZero = Theta2;
% remove bias column i.e. first column
Theta1_BiasZero(:,1) = 0;
Theta2_BiasZero(:,1) = 0;

% ====================================
% implementation detail: regularization of Theta matrices
% we could also do the following instead of above. Here we remove the bias 
% column instead of setting it to 0.
Theta1_unbiased = Theta1;
Theta2_unbiased = Theta2;
% remove bias column i.e. first column
Theta1_unbiased(:,1) = [];
Theta2_unbiased(:,1) = [];
% however in this method,the dimensions change - Theta1_unbiased is one column 
% less. This is OK for the reg term computation for cost (where we do element
% -wise squaring of matrices, but it is not OK for the reg term computation of 
% gradients where we do matrix addition (hence dimensions need to be maintained.

% implementation detail: back propagation with Theta matrices
% note - the unbiased Theta matrice will be used in the back propagation process
% just as in forward propagation bias nodes are added to the layers, in back
% propagation, bias columns are deducted from the matrices. Here we don't set to
% 0, we remove the column itself.
% ====================================


%a2 = sigmoid(z2)
z2 = zeros(m, hidden_layer_size); % i.e 5000 x 25 matrix
a2 = zeros(m, hidden_layer_size); % i.e 5000 x 25 matrix

%a3 = sigmoid(z3)
z3 = zeros(m, num_labels); % i.e 5000 x 10 matrix
a3 = zeros(m, num_labels); % i.e 5000 x 10 matrix


X = [ones(size(X,1),1) X]; % i.e 5000 x 401 matrix

% ==========================================================
% forward pass / forward propagation
% calculating the activations - a1,a2,a3 where a1 is input layer and 
% a3 is output layer has no associated error 

% z2 = Theta1' * a1 where a1 is X
% a2 = sigmoid (z2) plus bias column
z2 = X * Theta1'; % i.e 5000 x 25 matrix
a2 = sigmoid (z2); % i.e 5000 x 25 matrix

% adding bias
a2 = [ones(size(a2,1),1) a2];  
% now a2 is [m x (hidden_layer_size + 1)] i.e. 5000 x 26

% z3 = Theta2' * a2;
% a3 = sigmoid (z3) (no bias added to a3, a3 is output column)
z3 = a2 * Theta2'; % i.e 5000 x 10 matrix
a3 = sigmoid (z3); % i.e 5000 x 10 matrix

% forward pass end
% ==========================================================

% ==========================================================
% back propagation

% calculating the errors/gradients for every node in every layer  except ...
% input layer 
% in this example (a3 = HThetaX, a2 is hidden layer and a1 or X is input layer)
% in back propagation we compute d2,d3,delta2,delta1 and D

% d3 = a3 - y3 where d3 is error at output layer
d3 = a3 - y_matrix; 
% all three are (m x num_labels) i.e. 5000 x 10

% d2 = (Theta2' * d3) .* g_prime(z2) 
% where d2 is error at hidden layer with bias node removed since bias node...
% has no inputs, so remove bias column in Theta2 
d2 = (d3 * Theta2_unbiased) .* sigmoidGradient(z2);
% keeping in mind dimensions
% d3 is 5000x10
% Theta2 is 10x26, Theta2_unbiased is 10x25
% z2 is 5000x25
% hence result d2 is 5000x25

% delta2 = d3 * a2'; where a2 is the hidden layer
delta2 = d3' * a2;
% keeping in mind dimensions
% d3 is 5000x10
% a2 is 5000x26 (a2 has bias)
% hence result delta2 is 10x26; note this is same dimensions as Theta2

% delta1 = d2 * a1'; where a1 is the input layer X
delta1 = d2' * X;
% keeping in mind dimensions
% d2 is 5000x25
% X is 5000x401 (a1 has bias)
% hence result delta1 is 25x401; note this is same dimensions as Theta1


% unregularized gradient; see below for regularized gradients
% D1 = (1/m) * delta1;
% D2 = (1/m) * delta2;

% Theta1_grad = D1;
% Theta2_grad = D2;
% unregularized gradient end

% back propagation (without regularization) end
% ==========================================================



% ==========================================================
% cost computation

HThetaXik = a3;
% HThetaXik is [m x num_labels] i.e. 5000 x 10

YikLogHThetaXik = y_matrix .* log(HThetaXik);

One_HThetaXik = ones(m,num_labels) - HThetaXik;
One_YikLog1_HThetaXik = (ones(m,num_labels) - y_matrix) .* log(One_HThetaXik);

% cost J = (without regularization)
% J = sum (sum (-(1/m) * (YikLogHThetaXik + One_YikLog1_HThetaXik)));



% ================================================================
% regularization code
% ================================================================

% regularization term =
% (lambda/2m) * sum_across_all_layers [Thetas].^2 ignoring bias terms i.e
% the first columns of all the theta matrices - either remove them or set them 
% to zero,here the dimensions don't matter. see the impl note for 
% Theta1_BiasZero on top

RegTerm = 0;

% RegTerm = (1/2)*(1/m) * lambda * (sum(sum(Theta1_unbiased.^2)) + ...
%                                  sum(sum(Theta2_unbiased.^2)));
RegTerm = (1/2)*(1/m) * lambda * (sum(sum(Theta1_BiasZero.^2)) + ...
                                  sum(sum(Theta2_BiasZero.^2)));
% -------------------------------------------------------------

% cost J = (with regularization)
J = sum (sum (-(1/m) * (YikLogHThetaXik + One_YikLog1_HThetaXik))) + RegTerm;

% =========================================================================



% =========================================================================
% regularized gradients

% gradient with regularization =
% D'l' = (1/m) * delta'l'                         when j = 0; 
% D'l' = (1/m) * delta'l' + (lambda/m) * Theta'l' when j >= 1; 
% i.e. bias theta is not regularized, first column of theta is set to 0 
% (see similar process in multi class exercise 3.pdf and code, that was a ...
% vector theta, this is a matrix theta)

% gradients:
D1 = ((1/m) * delta1) + ((lambda/m) * Theta1_BiasZero);
% D1 is 25x401 - since delta1 is 25x401, same as Theta1, same as Theta1_BiasZero
D2 = ((1/m) * delta2) + ((lambda/m) * Theta2_BiasZero);
% D2 is 10x26 - since delta2 is 10x26, same as Theta2, same as Theta2_BiasZero

Theta1_grad = D1;
Theta2_grad = D2;

% regularized gradient end
% =========================================================================


% Unroll gradients into vectors
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
