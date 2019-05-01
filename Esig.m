function res = Esig(mu,sigma_sq)
kappa = (1+pi/8.*sigma_sq);
res = 1./(1+exp(-kappa.*mu));
