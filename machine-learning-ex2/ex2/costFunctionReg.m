function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculate H_theta(x)
h_theta = sigmoid( X * theta );

% Return all vector elements except theta_0 (theta_1)
_theta = theta(2:end);

% Remove first element of vector theta
norm_theta = [0;_theta];

% Calculate the Cost Function J
J = (1/m) * sum ( -y' * log( h_theta ) - (1 - y)' * log (1 - h_theta) ) + lambda * (1 / (2 * m) ) * norm_theta' * norm_theta;

% Calculate Gradient
% X(:,1) returns first Column of X matrix
% X(:,2:end) returns from second column up to the end of X matrix.
grad(1) = (1/m) *( h_theta - y )' * X(:,1);
grad(2:end) = ( (1/m) *( h_theta - y )'  * X(:,2:end) ) + ( (lambda / m ) * theta(2:end)' );


% =============================================================

end
