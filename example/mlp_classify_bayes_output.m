% mlp_classify_bayes_output
% for outputing var' estimate on toy 1d dataset

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
%function [c, posterior] = mlp_classify_bayes_output(M, x0, my, Q0, raw)
function mlp_classify_bayes_output(id)

file_id = sprintf('lapnn%s',id);
load(file_id);
x0 = linspace(-10,10,500)';

Q0 = [];
raw = 0;

alpha = 4-2*sqrt(2);
beta = -log(sqrt(2)+1);
lambda = sqrt(pi/8);

layers = M.structure.layers;
n_layers = length(layers);

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
            %if l==n_layers
            %    bingomax = max(posterioro_s)
            %    bingomin = min(posterioro_s)
            %end
        end

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
            end
        end

        if l == n_layers && M.output.binary
            %posterior = softmax(posterior);
            posterior = sigmoid(kappa(posterioro_s).*posterioro);
            posterior_s = sigmoid(kappa(posterioro_s,alpha)...
                .*(alpha*(posterioro+beta)),M.hidden.use_tanh)...
                -posterior.*posterior;
            %bingos = posterior_s(1:10,1:10)
            %bingo = posterior(1:10,1:10)
        end
    end
end

x = x0;
m = posterioro;
s = posterioro_s;
save(sprintf('%s_output.mat',file_id),'x','m','s');
