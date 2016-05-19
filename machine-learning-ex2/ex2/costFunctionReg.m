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

%theta_zero = theta(1)

%size(theta,2)

%theta = theta(2:size(theta,2))

%h_theta_zero = sigmoid( X * theta_zero )

h_theta = sigmoid( X * theta );

new_theta = theta(2:size(theta));

%size(new_theta)

temp = [0;new_theta];

temp' * temp

%size(temp)

%Cost Function J
%J = (1/m) * sum ( -y' * log( h_theta ) - (1 - y)' * log (1 - h_theta) ) + lambda * (1 / (2 * m) ) * sum (theta .^ 2);

J = (1/m) * sum ( -y' * log( h_theta ) - (1 - y)' * log (1 - h_theta) ) + lambda * (1 / (2 * m) ) * temp' * temp;

%theta

%(1 /m) * sum(sigmoid( X * theta ) - y) * X(:,1)'

grad(1) = (1/m) *( h_theta - y )'  * X(:,1);

(1/m) *( h_theta - y )'  * X + ( (lambda / m ) * theta )

 


#h_theta - y
#size(y)
#size(X(:,1))

#X(:,1)' * y

%(1 / m) * sum(sigmoid( X * theta ) - y) * X(2,2)

%(1 / m) * sum(sigmoid( X * theta ) - y) * X(2,3)

%(1 / m) * sum(sigmoid( X * theta ) - y) * X(2,:)

%grad(2:size(theta)





% =============================================================

end
