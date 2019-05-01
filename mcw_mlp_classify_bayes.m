% mlp_classify_bayes
% Copyright (C) 2015 Hao Wang
% Adapted from code by KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [c, posterior] = mcw_mlp_classify_bayes(M, x0, t0, my, Q0, raw)
rand('seed',11112);

if nargin < 5
    Q0 = [];
end

if nargin < 6
    raw = 0;
end

alpha = 4-2*sqrt(2);
beta = -log(sqrt(2)+1);
lambda = sqrt(pi/8);

layers = M.structure.layers;
n_layers = length(layers);


sum_posterior = zeros(size(x0,1),layers(end));
sum_posterior_s = zeros(size(x0,1),layers(end));

if isfield(M, 'dbm') && M.dbm.use % this option is not supported for Bayes' version
    for l = 2:n_layers
        if M.dropout.use && l > 2
            posterior = posterior * bsxfun(@times, (M.W{l-1}), 1 - M.dropout.probs{l-1});
        else
            posterior = posterior * M.W{l-1};
        end

        if l < n_layers-1
            posterior = posterior + Q0{l+1} * (M.dbm.W{l})';
        end
        posterior = bsxfun(@plus, posterior, M.biases{l}');

        if l < n_layers 
            posterior = sigmoid(posterior, M.hidden.use_tanh);
        end

        if l == n_layers && M.output.binary
            posterior = softmax(posterior);
        end
    end
else
    for i = 1:my.mc_weight
        posterioro = x0;
        posterior = x0;
        for l = 2:n_layers
            W = randn(size(M.W{l-1})).*sqrt(M.W_s{l-1})+M.W{l-1};
            biases = randn(size(M.biases{l})).*sqrt(M.biases_s{l})+M.biases{l};
            posterioro = bsxfun(@plus, posterior * W, biases');

            if l < n_layers 
                posterior = sigmoid(posterioro, M.hidden.use_tanh);

                % dropout
                if my.dropout~=0
                    mask = single(bsxfun(@minus,rand(size(posterior)),...
                        my.dropout*ones(1,size(posterior,2)))>0);
                    posterior = mask.*posterior;
                end
            end

            if l == n_layers && M.output.binary
                posterior = sigmoid(posterioro);
            end
        end
        if M.output.binary
            sum_posterior = sum_posterior+posterior;
            [maxp, c] = max(sum_posterior/i, [], 2);
            n_correct = sum(t0==c);
            fprintf('mc %d: %d\n',i,n_correct);
        else
            sum_posterior = sum_posterior+posterioro;
            sum_posterior_s = sum_posterior_s+posterioro.^2;
        end
    end
end

if raw
    c = posterior;
elseif my.regression_target==0
    posterior = sum_posterior/my.mc_dropout;
    [maxp, c] = max(posterior, [], 2);
else
    c = sum_posterior/my.mc_weight;
    posterior = sum_posterior_s/my.mc_weight;
end


