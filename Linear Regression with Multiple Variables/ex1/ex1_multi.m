%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 1;
num_iters = 400;


% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
%===========================================
%theta1 = zeros(3, 1);
%[theta1, J_history1] = gradientDescentMulti(X, y, theta1, 0.0001, num_iters);
%theta2 = zeros(3, 1);
%[theta2, J_history2] = gradientDescentMulti(X, y, theta2, 0.001, num_iters);
%theta3 = zeros(3, 1);
%[theta3, J_history3] = gradientDescentMulti(X, y, theta3, 0.002, num_iters);
%theta4 = zeros(3, 1);
%[theta4, J_history4] = gradientDescentMulti(X, y, theta4, 0.004, num_iters);
%theta5 = zeros(3, 1);
%[theta5, J_history5] = gradientDescentMulti(X, y, theta5, 0.008, num_iters);
%===========================================
%theta6 = zeros(3, 1);
%[theta6, J_history6] = gradientDescentMulti(X, y, theta6, 0.02, num_iters);
%theta7 = zeros(3, 1);
%[theta7, J_history7] = gradientDescentMulti(X, y, theta7, 0.04, num_iters);
%theta8 = zeros(3, 1);
%[theta8, J_history8] = gradientDescentMulti(X, y, theta8, 0.08, num_iters);
%theta9 = zeros(3, 1);
%[theta9, J_history9] = gradientDescentMulti(X, y, theta9, 0.16, num_iters);
%theta10 = zeros(3, 1);
%[theta10, J_history10] = gradientDescentMulti(X, y, theta10, 0.32, num_iters);
%theta11 = zeros(3, 1);
%[theta11, J_history11] = gradientDescentMulti(X, y, theta11, 0.64, num_iters);
%===========================================
%theta12 = zeros(3, 1);
%[theta12, J_history12] = gradientDescentMulti(X, y, theta12, 1.0, num_iters);
%theta13 = zeros(3, 1);
%[theta13, J_history13] = gradientDescentMulti(X, y, theta13, 1.1, num_iters);
%theta14 = zeros(3, 1);
%[theta14, J_history14] = gradientDescentMulti(X, y, theta14, 1.2, num_iters);
%theta15 = zeros(3, 1);
%[theta15, J_history15] = gradientDescentMulti(X, y, theta15, 1.25, num_iters);
%theta16 = zeros(3, 1);
%[theta16, J_history16] = gradientDescentMulti(X, y, theta16, 1.27, num_iters);
%theta17 = zeros(3, 1);
%[theta17, J_history17] = gradientDescentMulti(X, y, theta17, 1.28, num_iters);
%theta18 = zeros(3, 1);
%[theta18, J_history18] = gradientDescentMulti(X, y, theta18, 1.29, num_iters);
%theta19 = zeros(3, 1);
%[theta19, J_history19] = gradientDescentMulti(X, y, theta19, 1.30, num_iters);
%theta20 = zeros(3, 1);
%[theta10, J_history20] = gradientDescentMulti(X, y, theta20, 1.31, num_iters);
%theta21 = zeros(3, 1);
%[theta21, J_history21] = gradientDescentMulti(X, y, theta21, 1.32, num_iters);
%===========================================

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-k', 'LineWidth', 1);
%hold on;

%plot(1:numel(J_history1), J_history1, '-g', 'LineWidth', 1);
%plot(1:numel(J_history2), J_history2, '-g', 'LineWidth', 1);
%plot(1:numel(J_history3), J_history3, '-g', 'LineWidth', 1);
%plot(1:numel(J_history4), J_history4, '-g', 'LineWidth', 1);
%plot(1:numel(J_history5), J_history5, '-g', 'LineWidth', 1);

%plot(1:numel(J_history6), J_history6, '-r', 'LineWidth', 1);
%plot(1:numel(J_history7), J_history7, '-r', 'LineWidth', 1);
%plot(1:numel(J_history8), J_history8, '-r', 'LineWidth', 1);
%plot(1:numel(J_history9), J_history9, '-r', 'LineWidth', 1);
%plot(1:numel(J_history10), J_history10, '-r', 'LineWidth', 1);
%plot(1:numel(J_history11), J_history11, '-r', 'LineWidth', 1);

%plot(1:numel(J_history12), J_history12, '-b', 'LineWidth', 1);
%plot(1:numel(J_history13), J_history13, '-b', 'LineWidth', 1);
%plot(1:numel(J_history14), J_history14, '-b', 'LineWidth', 1);
%plot(1:numel(J_history15), J_history15, '-b', 'LineWidth', 1);
%plot(1:numel(J_history16), J_history16, '-b', 'LineWidth', 1);
%plot(1:numel(J_history17), J_history17, '-b', 'LineWidth', 1);
%plot(1:numel(J_history18), J_history18, '-b', 'LineWidth', 1);
%plot(1:numel(J_history19), J_history19, '-b', 'LineWidth', 1);
%plot(1:numel(J_history20), J_history20, '-b', 'LineWidth', 1);
%plot(1:numel(1:50), J_history21(1:50), '-b', 'LineWidth', 1);

xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

price = [1650 3];
price = (price - mu) ./ sigma;
price = [1, price(1), price(2)];
price *= theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

price = [1650 3];
price = [1, price(1), price(2)];
price *= theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

