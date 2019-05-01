% mlp_classify_bayes_output_bb
% for outputing var' estimate on plate dataset

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
function [SE, VAR] = mlp_classify_bayes_output_bb(id)

load(sprintf('lapnn%d.mat',id));
load bb_regression;
x0 = X_test;
t0 = X_test_labels;

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

SE = sum((posterioro-t0).^2,2);
VAR = sum(posterioro_s,2);
save(sprintf('lapnn%d_SE_VAR.mat',id),'VAR','SE');

if my.print_wrongs==1
    myfid = fopen(sprintf('%s.wrongs',my.save),'w');
    fprintf(myfid,'true labels:\n');
    dlmwrite(sprintf('%s.wrongs',my.save),[find(t0~=c) t0(t0~=c)],'-append');
    fclose(myfid);
    myfid = fopen(sprintf('%s.wrongs',my.save),'a');
    fprintf(myfid,'predictions:\n');
    dlmwrite(sprintf('%s.wrongs',my.save),c(t0~=c),'-append');
    fclose(myfid);
    myfid = fopen(sprintf('%s.wrongs',my.save),'a');
    fprintf(myfid,'means:\n');
    dlmwrite(sprintf('%s.wrongs',my.save),posterior(t0~=c,:),'-append');
    fclose(myfid);
    myfid = fopen(sprintf('%s.wrongs',my.save),'a');
    fprintf(myfid,'variances:\n');
    dlmwrite(sprintf('%s.wrongs',my.save),posterior_s(t0~=c,:),'-append');
    fclose(myfid);
end
if my.print_rights==1
    myfid = fopen(sprintf('%s.rights',my.save),'w');
    fprintf(myfid,'true labels:\n');
    dlmwrite(sprintf('%s.rights',my.save),t0(t0==c),'-append');
    fclose(myfid);
    myfid = fopen(sprintf('%s.rights',my.save),'a');
    fprintf(myfid,'means:\n');
    dlmwrite(sprintf('%s.rights',my.save),posterior(t0==c,:),'-append');
    fclose(myfid);
    myfid = fopen(sprintf('%s.rights',my.save),'a');
    fprintf(myfid,'variances:\n');
    dlmwrite(sprintf('%s.rights',my.save),posterior_s(t0==c,:),'-append');
    fclose(myfid);
end
if my.use_second==1
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
    v_scores = find(scores>=7)'
    posterior2 = posterior;
    posterior2(sub2ind(size(posterior),1:size(posterior,1),c')) = -inf;
    [maxp2 c2] = max(posterior2,[],2);
    num_changes = size(find(c(v_scores)~=c2(v_scores)))
    c(v_scores) = c2(v_scores);
    bingom = posterior(v_scores,:)'
    bingos = posterior_s(v_scores,:)'
end
