% simple wrapper for kappa function
function [y] = kappa(x, alpha, const)
if nargin < 3
    const = 1;
end
if nargin < 2
    alpha = 1;
end

y = 1./sqrt(const+pi/8*alpha*alpha*x);
