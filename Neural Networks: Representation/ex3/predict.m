function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

hidden_layer_size = size(Theta2, 2) - 1;

Y = zeros(m, hidden_layer_size);
Z = zeros(m, num_labels);
probability = zeros(m, 1);

X = [ones(m, 1), X];

Y = sigmoid(X * Theta1');

Y = [ones(m, 1), Y];

Z = sigmoid(Y * Theta2');

[probability, p] = max(Z, [], 2);

% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');

% [dummy, p] = max(h2, [], 2);

% =========================================================================


end
