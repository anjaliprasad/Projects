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

z = X * theta;
g=sigmoid(z);
thetaWithout=theta(2:end, :);
thetaForGrad=theta;
thetaForGrad(1)=0;
reg=lambda*(thetaWithout'*thetaWithout)/(2*m);
for i = 1:m
    J(i) = ((-1*y(i).*log(g(i))) - ((1-y(i)).*log(1-g(i))))+reg;
end
J=(1/m).* sum(J);
srt_error=(g - y);
grad=(1/m) * (srt_error' * X ) + ((lambda/m).*thetaForGrad)'; 


% =============================================================

end
