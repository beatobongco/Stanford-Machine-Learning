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

a1 = [ones(size(X, 1), 1) X]; % add new col of ones

% deconstructing a * theta'
% a and theta are both _ x n matrices where n is your num features
% you're taking each row of a, then applying each row of theta to it and summing the values
% each row of new matrix will be (r = row 1 of a)
% [sum(row 1 of theta .* r),  sum(row 2 of theta .* r), ...]

a2 = [ones(size(a1, 1), 1) sigmoid(a1 * Theta1')];
a3 = sigmoid(a2 * Theta2');

% max(a, [], 2) gets the max of each row and returns 2 vals, maxes and index
[M , p] = max(a3 , [] , 2);
% =========================================================================


end
