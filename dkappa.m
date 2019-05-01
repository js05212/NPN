% simple wrapper for derivative of kappa function
function [y] = dkappa(x, alpha, const)
if nargin < 3
    const = 1;
end

if nargin < 2
    alpha = 1;
end

y = -pi/16*alpha*alpha*(const+pi/8*alpha*alpha*x).^(-1.5);
