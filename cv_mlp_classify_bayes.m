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
function [c, posterior] = cv_mlp_classify_bayes(M, x0, t0, my, Q0, raw)

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
                %bingos1 = posterior(1:5,1:5)
                %bingos2 = posterior_s(1:5,1:5)
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
    end
end

[maxp, c] = max(posterior, [], 2);
c_ori = c;

maxi = 0;
maxj = 0;
maxk = 0;
cur_max = 0;
for i = 1:length(my.m_gaps)
    for j = 1:length(my.v_gaps)
        for k = 1:length(my.m_val)
            c = c_ori;
            my.use_second = 1;
            my.thre.first = 1.9999999;
            my.thre.second = my.m_val(k);
            my.thre.var_value1 = 0.00;
            my.thre.var_value2 = 0;
            my.thre.mean_gap = my.m_gaps(i);
            my.thre.var_gap = my.v_gaps(j);

            v_first = find(maxp<my.thre.first);

            posterior1 = posterior(v_first,:);
            posterior1(sub2ind(size(posterior1),1:length(v_first),c(v_first)')) = -inf;
            [maxp1 c1] = max(posterior1,[],2);
            v_second = find(maxp1>my.thre.second);
            v_second_abs = v_first(v_second);

            v_var_value1 = ...
                find(posterior_s(sub2ind(size(posterior_s),v_second_abs,c(v_second_abs)))>...
                my.thre.var_value1);
            v_var_value_abs1 = v_second_abs(v_var_value1);

            v_var_value2 = ...
                find(posterior_s(sub2ind(size(posterior_s),v_second_abs,c1(v_second)))>...
                my.thre.var_value2);
            v_var_value_abs2 = v_second_abs(v_var_value2);

            v_var_gap = ...
                find(posterior_s(sub2ind(size(posterior_s),v_second_abs,c1(v_second)))-...
                    posterior_s(sub2ind(size(posterior_s),v_second_abs,c(v_second_abs)))<...
                my.thre.var_gap);
            v_var_gap_abs = v_second_abs(v_var_gap);

            v_var_gap0 = ...
                find(posterior_s(sub2ind(size(posterior_s),v_second_abs,c1(v_second)))-...
                    posterior_s(sub2ind(size(posterior_s),v_second_abs,c(v_second_abs)))>...
                0);
            v_var_gap_abs0 = v_second_abs(v_var_gap0);

            v_mean_gap = ...
                find(posterior(sub2ind(size(posterior),v_second_abs,c1(v_second)))-...
                    posterior(sub2ind(size(posterior),v_second_abs,c(v_second_abs)))>...
                -my.thre.mean_gap);
            v_mean_gap_abs = v_second_abs(v_mean_gap);

            scores = zeros(size(c));
            scores(v_first) = scores(v_first)+1;
            scores(v_second_abs) = scores(v_second_abs)+1;
            scores(v_var_value_abs1) = scores(v_var_value_abs1)+1;
            scores(v_var_value_abs2) = scores(v_var_value_abs2)+1;
            scores(v_var_gap_abs) = scores(v_var_gap_abs)+1;
            scores(v_var_gap_abs0) = scores(v_var_gap_abs0)+1;
            scores(v_mean_gap_abs) = scores(v_mean_gap_abs)+1;
            v_scores = find(scores>=7)';
            posterior2 = posterior;
            posterior2(sub2ind(size(posterior),1:size(posterior,1),c')) = -inf;
            [maxp2 c2] = max(posterior2,[],2);
            num_changes = size(find(c(v_scores)~=c2(v_scores)));
            c(v_scores) = c2(v_scores);
            n_correct = sum(t0==c);
            my.acc(i,j) = n_correct;
            if n_correct>cur_max
                cur_max = n_correct;
                maxi = i;
                maxj = j;
                maxk = k;
            end
            fprintf(2,'mean/var/val/acc: %0.3f/%0.3f/%0.3f/%d, best:%0.3f/%0.3f/%0.3f/%d\n',...
                my.m_gaps(i),my.v_gaps(j),my.m_val(k),n_correct,my.m_gaps(maxi),...
                my.v_gaps(maxj),my.m_val(maxk),cur_max);
            myfid = fopen(sprintf('%s.cv',my.save),'a');
            fprintf(myfid,'mean/var/val/acc: %0.3f/%0.3f/%0.3f/%d, best:%0.3f/%0.3f/%0.3f/%d\n',...
                my.m_gaps(i),my.v_gaps(j),my.m_val(k),n_correct,my.m_gaps(maxi),...
                my.v_gaps(maxj),my.m_val(maxk),cur_max);
            fclose(myfid);
        end
    end
end
