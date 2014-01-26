function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]';
sigma_list = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 1.00, 30.0]';

C_size = size(C_list, 1);
sigma_size = size(sigma_list, 1);

list_size = C_size .* sigma_size;
list = zeros(C_size .* sigma_size, 3);

for i = 1:C_size
  for j = 1:sigma_size
    % get model
    model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
    % get predictions
    predictions = svmPredict(model, Xval);
    % get mean
    mean_this = mean(double(predictions ~= yval));
    % record
    list((i - 1) .* C_size + j, 1) = C_list(i);
    list((i - 1) .* C_size + j, 2) = sigma_list(j);
    list((i - 1) .* C_size + j, 3) = mean_this;
    % debug
    fprintf("\nlist(%d) is (\t%f, \t%f, \t%f).",
	    (i - 1) .* C_size + j, 
	    list((i - 1) .* C_size + j, 1),
	    list((i - 1) .* C_size + j, 2),
	    list((i - 1) .* C_size + j, 3));
  end
end

min = list(1, 3);
index = 1;

for i = 2:list_size
  if list(i, 3) < min
    min = list(i, 3);
    index = i;
  end
end

C = list(index, 1);
sigma = list(index, 2);

fprintf("\nThe index is %d\n", index);
fprintf("\nThe min C is %f, the min sigma is %f.", list(index, 1), list(index, 2));

% =========================================================================

end
