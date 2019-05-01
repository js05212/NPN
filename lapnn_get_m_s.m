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
function [m, s] = lapnn_get_m_s(M, x0, t0, my, Q0, raw)

if nargin < 5
    Q0 = [];
end

if nargin < 6
    raw = 0;
end

alpha = 4-2*sqrt(2);
beta = -log(sqrt(2)+1);
alphap = alpha*2;
betap = beta/2;
lambda = sqrt(pi/8);

layers = M.structure.layers;
n_layers = length(layers);

if isfield(M,'mean') && my.subtract_mean==1
    x0 = bsxfun(@minus, x0, M.mean);
end

posterioro = x0;
posterioro_s = zeros(size(x0));
posterior = x0;
posterior_s = zeros(size(x0));


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
    for l = 2:n_layers
        if M.dropout.use && l > 2
            posterioro = bsxfun(@plus, posterior * (M.W{l-1}*(1-my.dropout)),M.biases{l}'*(1-my.dropout));
            posterioro_s = bsxfun(@plus, ...
                posterior_s*M.W_s{l-1}*(1-my.dropout)...
                +(posterior.*posterior)*M.W_s{l-1}*(1-my.dropout)...
                +posterior_s*(M.W{l-1}.*M.W{l-1})*(1-my.dropout),...
                M.biases_s{l}'*(1-my.dropout)); % bayes, hog
        else
            posterioro = bsxfun(@plus, posterior * (M.W{l-1}), M.biases{l}');
            posterioro_s = bsxfun(@plus, ...
                posterior_s*M.W_s{l-1}...
                +(posterior.*posterior)*M.W_s{l-1}...
                +posterior_s*(M.W{l-1}.*M.W{l-1}),...
                M.biases_s{l}'); % bayes, hog
        end
        %bingo = posterioro(1:5,1:5)
        %bingos = posterioro_s(1:5,1:5)

        if l < n_layers 
            if my.use_tanh==0
                posterior = sigmoid(kappa(posterioro_s).*posterioro,...
                    M.hidden.use_tanh);
                posterior_s = sigmoid(kappa(posterioro_s,alpha)...
                    .*(alpha*(posterioro+beta)),M.hidden.use_tanh)...
                    -posterior.*posterior;
            elseif my.use_tanh==2
                ratio_v = posterioro./posterioro_s;
                posterior = sigmoid(ratio_v/lambda).*posterioro+sqrt(posterioro_s)/sqrt(2*pi).*exp(-0.5*ratio_v.^2);
                posterior_s = sigmoid(ratio_v/lambda).*(posterioro.^2+posterioro_s)+posterioro.*sqrt(posterioro_s)/sqrt(2*pi).*exp(-0.5*ratio_v.^2)-posterior.^2;
            elseif my.use_tanh==1
                posterior = 2*sigmoid(posterioro.*kappa(posterioro_s,1,0.25))-1;
                posterior_s = 4*sigmoid(alphap*(posterioro+betap).*...
                    kappa(posterioro_s,alphap))-4*sigmoid(posterioro...
                    .*kappa(posterioro_s,1,0.25))+1-posterior.^2;
            end
        end

        if l == n_layers && M.output.binary
            %posterior = softmax(posterior);
            posterior = sigmoid(kappa(posterioro_s).*posterioro);
            posterior_s = sigmoid(kappa(posterioro_s,alpha)...
                .*(alpha*(posterioro+beta)))...
                -posterior.*posterior;
            %pos = posterior;
            %posterior = sigmoid(kappa(posterior_s).*posterior);
            %posterior_s = sigmoid(kappa(posterior_s,alpha)...
            %    .*(alpha*(pos+beta)))...
            %    -posterior.*posterior;
        end
        %bingos1 = posterior(1:5,1:5)
        %bingos2 = posterior_s(1:5,1:5)
    end
end

m = posterior;
s = posterior_s;
