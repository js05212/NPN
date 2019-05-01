% mlp_bayes - training an Bayesian MLP
% Copyright (C) 2017 Hao Wang
% This is the core code for Gaussian NPN.

% This program is adapted from deepmat by KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [M] = mlp_bayes(M, patches, targets, X_test, X_test_labels, my,...
    patches_s, valid_patches, valid_targets, ...
    valid_portion, use_cvp)
if nargin < 7
    patches_s = [];
end

if nargin < 8 % modified by Hao from 3 to 8
    early_stop = 0;
    valid_patches = [];
    valid_targets = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end

if nargin < 10
    use_cvp = 0; % modified by Hao to 0
end

alpha = 4-2*sqrt(2);
beta = -log(sqrt(2)+1);
alphap = alpha*2;
betap = beta/2;
lambda_sq = pi/8;
lambda = sqrt(pi/8);
my.eps = 1e-6; % eps for gaussian_s

actual_lrate = M.learning.lrate;

n_samples = size(patches, 1);

layers = M.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = M.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

if use_cvp
    cvp = crossvalind('Kfold', targets, n_minibatches);
end

if size(targets, 2) == 1 && M.output.binary
    % populate the target labels
    % more efficient way to populate the target labels thanks to vikrantt
    C=unique(targets);
    new_targets = bsxfun(@eq,repmat(targets,1,length(C)),repmat(C',length(targets),1));
    targets = new_targets;
end

if size(valid_targets, 2) == 1 && M.output.binary
    % populate the target labels
    % more efficient way to populate the target labels thanks to vikrantt
    C=unique(valid_targets);
    new_targets = bsxfun(@eq,repmat(valid_targets,1,length(C)),repmat(C',length(valid_targets),1));
    valid_targets = new_targets;
end

n_epochs = M.iteration.n_epochs;

momentum = M.learning.momentum;
% weight decay for W and biases
weight_decay = M.learning.weight_decay;
% weight decay for W_s and biases_s
weight_decay_s = M.learning.weight_decay_s;

biases_grad = cell(n_layers, 1);
W_grad = cell(n_layers, 1);
biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
% init' XX_s variables for NPN besides the original ones
piases_s_grad = cell(n_layers, 1);
M_s_grad = cell(n_layers, 1);
piases_s_grad_old = cell(n_layers, 1);
M_s_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad{l} = zeros(size(M.biases{l}))';
    piases_s_grad{l} = zeros(size(M.piases_s{l}))';
    if l < n_layers
        W_grad{l} = zeros(size(M.W{l}));
        M_s_grad{l} = zeros(size(M.M_s{l}));
    end
    biases_grad_old{l} = zeros(size(M.biases{l}))';
    piases_s_grad_old{l} = zeros(size(M.piases_s{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(M.W{l}));
        M_s_grad_old{l} = zeros(size(M.M_s{l}));
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = M.do_normalize;
do_normalize_std = M.do_normalize_std;

% not used for MNIST experiments
if M.data.binary == 0
    if do_normalize == 1
        % make it zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
        M.mean = patches_mean;
    end

    if do_normalize_std ==1
        % make it unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
        M.std = patches_std;
    end
end

% not used for MNIST experiments
if my.subtract_mean==1
    % make it zero-mean
    patches_mean = mean(patches, 1);
    %patches_mean = ones(1,size(patches,2))*0.5;
    patches = bsxfun(@minus, patches, patches_mean);
    M.mean = patches_mean;
end

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if M.debug.do_display == 1
    figure(M.debug.display_fid);
end

try
    use_gpu = gpuDeviceCount;
catch errgpu
    use_gpu = false;
    disp(['Could not use CUDA. Error: ' errgpu.identifier])
end
for step=my.from:n_epochs
    if M.verbose
        myfid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(myfid, 'Epoch %d/%d: ', step, n_epochs)
        fclose(myfid);
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        for l = 1:n_layers
            if l < n_layers 
                M.W{l} = gpuArray(single(M.W{l}));
                M.W_s{l} = gpuArray(single(M.W_s{l}));
                M.M_s{l} = gpuArray(single(M.M_s{l}));
            end
            M.biases{l} = gpuArray(single(M.biases{l}));
            M.biases_s{l} = gpuArray(single(M.biases_s{l}));
            M.piases_s{l} = gpuArray(single(M.piases_s{l}));
        end

        if M.adagrad.use 
            for l = 1:n_layers
                if l < n_layers 
                    M.adagrad.W{l} = gpuArray(single(M.adagrad.W{l}));
                    M.adagrad.M_s{l} = gpuArray(single(M.adagrad.M_s{l}));
                end
                M.adagrad.biases{l} = gpuArray(single(M.adagrad.biases{l}));
                M.adagrad.piases_s{l} = gpuArray(single(M.adagrad.piases_s{l}));
            end
        elseif M.adadelta.use
            for l = 1:n_layers
                if l < n_layers 
                    M.adadelta.gW{l} = gpuArray(single(M.adadelta.gW{l}));
                    M.adadelta.W{l} = gpuArray(single(M.adadelta.W{l}));
                    M.adadelta.gM_s{l} = gpuArray(single(M.adadelta.gM_s{l}));
                    M.adadelta.M_s{l} = gpuArray(single(M.adadelta.M_s{l}));
                end
                M.adadelta.gbiases{l} = gpuArray(single(M.adadelta.gbiases{l}));
                M.adadelta.biases{l} = gpuArray(single(M.adadelta.biases{l}));
                M.adadelta.gpiases_s{l} = gpuArray(single(M.adadelta.gpiases_s{l}));
                M.adadelta.piases_s{l} = gpuArray(single(M.adadelta.piases_s{l}));
            end
        end
    end

    for mb=1:n_minibatches
        M.iteration.n_updates = M.iteration.n_updates + 1;

        if use_cvp
            v0 = patches(cvp == mb, :);
            t0 = targets(cvp == mb, :);
            if my.init_s==0
                v0s = zeros(size(v0)); % NPN, Hao
            else
                v0s = ones(size(v0))*my.init_s;
            end
        else
            mb_start = (mb - 1) * minibatch_sz + 1;
            mb_end = min(mb * minibatch_sz, n_samples);

            % p_0
            v0 = patches(mb_start:mb_end, :);
            t0 = targets(mb_start:mb_end, :);
            if size(patches_s,1)~=0
                v0s = patches_s(mb_start:mb_end,:);
            else
                v0s = zeros(size(v0)); % NPN, Hao
            end

            if my.init_s~=0
                v0s = v0s+ones(size(v0s))*my.init_s;
            end
        end
        mb_sz = size(v0,1);


        if use_gpu > 0
            v0 = gpuArray(single(v0));
            v0s = gpuArray(single(v0s)); % NPN, Hao
        end

        % add error
        v0_clean = v0;

        if M.data.binary == 0 && M.noise.level > 0
            v0 = v0 + M.noise.level * gpuArray(randn(size(v0)));
        end

        % denoising
        % 1: match mean/var' of masking noise
        % 2: independently mask mean and var'
        % 3: mask noise for mean only
        % 4: same mask noise for mean and var'
        % 5: init' var' according to the current batch
        if M.noise.drop > 0
            if my.denoising_type==1
                v0 = v0 * (1-M.noise.drop);
                v0s = v0.*v0*M.noise.drop*(1-M.noise.drop);
            elseif my.denoising_type==2
                mask = binornd(1, 1 - M.noise.drop, size(v0));
                mask_s = binornd(1, 1 - M.noise.drop, size(v0));
                v0 = v0.*mask;
                v0s = v0s.*mask_s;
                clear mask;
            elseif my.denoising_type==3
                v0 = v0 + M.noise.drop * gpuArray(randn(size(v0)));
            elseif my.denoising_type==4
                mask = binornd(1, 1 - M.noise.drop, size(v0));
                v0 = v0.*mask;
                v0s = v0s.*mask;
                clear mask;
            elseif my.denoising_type==5
                v0_mean = mean(v0, 1);
                v0_centered = bsxfun(@minus, v0, v0_mean);
                v0_noise = diag(v0_centered'*v0_centered)/size(v0,1);
                v0s = repmat(v0_noise',size(v0,1),1);
            end
        end

        % init'
        h0 = cell(n_layers, 1);  % a_m, Hao
        h0s = cell(n_layers, 1); % a_s, Hao
        o0 = cell(n_layers, 1);  % o_m, Hao
        o0s = cell(n_layers, 1); % o_s, Hao
        h0mask = cell(n_layers, 1);
        h0{1} = v0;
        h0s{1} = v0s;
        o0{1} = v0;
        o0s{1} = v0s;

        % feedforward
        for l = 2:n_layers
            % linear NPN
            o0{l} = bsxfun(@plus, h0{l-1} * M.W{l-1}, M.biases{l}');
            o0s{l} = bsxfun(@plus, h0s{l-1}*M.W_s{l-1} ... 
                +(h0{l-1}.*h0{l-1})*M.W_s{l-1} ...
                +h0s{l-1}*(M.W{l-1}.*M.W{l-1}),M.biases_s{l}'); % NPN, Hao

            % nonlinear NPN
            if l < n_layers
                if my.use_tanh==0
                    h0{l} = sigmoid(kappa(o0s{l}).*o0{l}, M.hidden.use_tanh);
                    h0s{l} = sigmoid(kappa(o0s{l},alpha).*(alpha*(o0{l}+beta)),M.hidden.use_tanh)-h0{l}.*h0{l};
                elseif my.use_tanh==2
                    ratio_v = o0{l}./sqrt(o0s{l}); % tmp variable
                    h0{l} = sigmoid(ratio_v/lambda).*o0{l}+sqrt(o0s{l})/sqrt(2*pi).*exp(-0.5*ratio_v.^2);
                    h0s{l} = sigmoid(ratio_v/lambda).*(o0{l}.^2+o0s{l})+o0{l}.*sqrt(o0s{l})/sqrt(2*pi).*exp(-0.5*ratio_v.^2)-h0{l}.^2;

                    clear ratio_v;
                elseif my.use_tanh==1
                    h0{l} = 2*sigmoid(o0{l}.*kappa(o0s{l},1,0.25))-1;
                    h0s{l} = 4*sigmoid(alphap*(o0{l}+betap).*kappa(o0s{l},alphap))-4*sigmoid(o0{l}.*kappa(o0s{l},1,0.25))+1-h0{l}.^2;
                end
            end

            % dropout (same mask for mean and var')
            if M.dropout.use && l < n_layers
                h0mask{l} = single(bsxfun(@minus, rand(size(h0{l})), ...
                    M.dropout.probs{l}') > 0);
                h0{l} = h0mask{l} .* h0{l};
                h0s{l} = h0mask{l} .* h0s{l};
            end

            if l == n_layers && M.output.binary
                % for the last layer, only need h0 as prediction
                h0{l} = sigmoid(kappa(o0s{l}).*o0{l});
                h0s{l} = sigmoid(kappa(o0s{l},alpha).*(alpha*(o0{l}+beta)))-h0{l}.*h0{l};
            end
        end

        % reset gradients
        for l = 1:n_layers
            biases_grad{l} = 0 * biases_grad{l};
            piases_s_grad{l} = 0 * piases_s_grad{l};
            if l < n_layers
                W_grad{l} = 0 * W_grad{l};
                M_s_grad{l} = 0 * M_s_grad{l};
            end
        end

        % backprop
        delta = cell(n_layers, 1); % gradient of a_m
        deltao = cell(n_layers, 1); % gradient of o_m
        delta_s = cell(n_layers, 1); % gradient of a_s
        deltao_s = cell(n_layers, 1); % gradient of o_s

        if M.output.binary==1
            % Gaussian loss as extra regulerization for MNIST with small training sets
            if my.gaussian_s~=0 
                deltao{end} = ...
                    0.5./(h0s{end}+my.eps).*(1-(h0{end}-t0).^2./(h0s{end}+my.eps)).*(alpha*dsigmoid(h0s{end}+h0{end}.^2).*kappa(o0s{end},alpha)...
                    -2*h0{end}.*dsigmoid(h0{end}).*kappa(o0s{end}))+dsigmoid(h0{end}).*kappa(o0s{end}).*(h0{end}-t0)./(h0s{end}+my.eps);
                deltao_s{end} = ...
                    0.5./(h0s{end}+my.eps).*(1-(h0{end}-t0).^2./(h0s{end}+my.eps)).*...
                    (dsigmoid(h0s{end}+h0{end}.^2).*(alpha*(o0{end}+beta)).*dkappa(o0s{end},alpha)...
                    -2*h0{end}.*dsigmoid(h0{end}).*o0{end}.*dkappa(o0s{end}))+dsigmoid(h0{end}).*o0{end}.*dkappa(o0s{end}).*(h0{end}-t0)./...
                    (h0s{end}+my.eps);
            else % cross-entropy loss
                deltao{end} = (h0{end} - t0).*kappa(o0s{end}); % modified by Hao
                deltao_s{end} = (h0{end} - t0).*o0{end}.*dkappa(o0s{end}); % NPN, Hao
            end
        else % for regression tasks
            % Gaussian loss (simplified KL loss)
            if my.regression_target==1 
                deltao{end} = 1./o0s{end}.*(o0{end}-t0);
                deltao_s{end} = 0.5./o0s{end}-0.5./o0s{end}.^2.*(o0{end}-t0).^2;
            else % L2 loss
                deltao{end} = o0{end}-t0;
                deltao_s{end} = 0 * t0;
            end
            % L2 regulerization for o_s
            if my.output_s~=0
                deltao_s{end} = deltao_s{end}+my.output_s*o0s{end};
            end
        end

        % BP
        % my.output_s.binary=1 for the MNIST task
        % my.output_s is the coefficient of a_s' L2 regulerization
        if my.output_s~=0 && M.output.binary==1
            t1 = t0;
            if my.opposite_s~=0
                t1(t1==0) = -1*my.opposite_s;
            end
            deltao{end} = deltao{end}+my.output_s*t1.*h0s{end}...
                .*(alpha*dsigmoid(h0s{end}+h0{end}.*h0{end})...
                -2*h0{end}.*dsigmoid(h0{end}).*kappa(o0s{end}));
            deltao_s{end} = deltao_s{end}+my.output_s*t1.*h0s{end}...
                .*(dsigmoid(h0s{end}+h0{end}.*h0{end}).*...
                (alpha*(o0{end}+beta)).*dkappa(o0s{end},alpha)...
                -2*h0{end}.*dsigmoid(h0{end}).*o0{end}.*dkappa(o0s{end}));
        end

        if M.output.binary
            if use_cvp
                xt = targets(cvp == mb, :);
            else
                xt = targets(mb_start:mb_end, :);
            end
            %if my.gaussian_s==0
                rerr = -mean(sum(xt .* log(max(h0{end}, 1e-16)) + (1 - xt) .* log(max(1 - h0{end}, 1e-16)), 2));
            %else
                %rerr = mean(sum(0.5.*log(h0s{end}+my.eps)+0.5./h0s{end}.*(h0{end}-t0).^2,2));
                %if mb==1
                %    fprintf('bingos %f\n',h0s{end}(1,1));
                %end
            %end
        else
            rerr = sqrt(mean(sum((o0{end}-t0).^2,2))); % RMSE
            % log-likelihood
            train_LL = mean(-0.5./o0s{end}.*(o0{end}-t0).^2-...
                0.5*log(2*pi*o0s{end}));
        end
        % gather evaluation statistics
        if use_gpu > 0
            rerr = gather(rerr);
            if my.regression_target~=0 
                train_LL = gather(train_LL);
            end
        end
        if my.regression_target~=0 
            M.signals.recon_errors = [M.signals.recon_errors rerr];
            M.signals.train_LL = [M.signals.train_LL train_LL];
        end

        % update biases (b_m and b_s)
        biases_grad{end} = mean(deltao{end}, 1);
        if my.positive_s==1
            piases_s_grad{end} = mean(deltao_s{end}, 1).*...
                exp(M.piases_s{end}')./(1+exp(M.piases_s{end}'));
        else
            piases_s_grad{end} = mean(deltao_s{end}, 1);
        end

        % BP through every layer
        for l = n_layers-1:-1:1
            delta{l} = deltao{l+1} * M.W{l}'+2*(deltao_s{l+1}*M.W_s{l}').*...
                h0{l};
            delta_s{l} = deltao_s{l+1}*M.W_s{l}'+deltao_s{l+1}*(M.W{l}.*M.W{l})';
            if my.deep_s~=0
                delta_s{l} = delta_s{l}+my.deep_s*h0s{l};
            end
            if l == 1 && M.data.binary
                deltao{l} = delta{l} .* dsigmoid(h0{l}).*kappa(o0s{l})...
                    +alpha*delta_s{l}.*dsigmoid(h0s{l}+h0{l}.*h0{l})...
                    -2*h0{l}.*delta_s{l}.*dsigmoid(h0{l}).*kappa(o0s{l});
                deltao_s{l} = ...
                    delta{l}.*dsigmoid(h0{l}).*o0{l}.*dkappa(o0s{l})...
                    +delta_s{l}.*dsigmoid(h0s{l}+h0{l}.*h0{l}).*...
                    (alpha*(o0{l}+beta)).*dkappa(o0s{l},alpha)...
                    -2*h0{l}.*delta_s{l}.*dsigmoid(h0{l}).*o0{l}.*...
                    dkappa(o0s{l});
            end
            if l > 1
                if my.use_tanh==0 % sigmoid
                    deltao{l} = delta{l} .* dsigmoid(h0{l},M.hidden.use_tanh).*kappa(o0s{l})...
                        +alpha*delta_s{l}.*dsigmoid(h0s{l}+h0{l}.*h0{l},M.hidden.use_tanh)...
                        -2*h0{l}.*delta_s{l}.*dsigmoid(h0{l},M.hidden.use_tanh).*kappa(o0s{l});
                    deltao_s{l} = ...
                        delta{l}.*dsigmoid(h0{l},M.hidden.use_tanh).*o0{l}.*dkappa(o0s{l})...
                        +delta_s{l}.*dsigmoid(h0s{l}+h0{l}.*h0{l},M.hidden.use_tanh).*...
                        (alpha*(o0{l}+beta)).*dkappa(o0s{l},alpha)...
                        -2*h0{l}.*delta_s{l}.*dsigmoid(h0{l},M.hidden.use_tanh).*o0{l}.*...
                        dkappa(o0s{l});
                elseif my.use_tanh==2 % ReLU
                    ratio_v = o0{l}./sqrt(o0s{l});
                    sigmoid_v = sigmoid(ratio_v/lambda);
                    exp_sq = exp(-0.5*ratio_v.^2);
                    part_mm = dsigmoid(sigmoid_v).*ratio_v/lambda+sigmoid_v-1/sqrt(2*pi)*ratio_v.*exp_sq;
                    part_ms = -0.5/lambda*ratio_v.^2./sqrt(o0s{l}).*dsigmoid(sigmoid_v)...
                        +exp_sq/sqrt(2*pi).*(0.5./sqrt(o0s{l})+0.5*ratio_v.^2./sqrt(o0s{l}));

                    deltao{l} = delta{l}.*part_mm+delta_s{l}.*(1/lambda*dsigmoid(sigmoid_v).*(ratio_v.*o0{l}+sqrt(o0s{l}))+2*sigmoid_v.*o0{l}...
                        +sqrt(o0s{l})/sqrt(2*pi).*exp_sq.*(1-ratio_v.^2)-2*h0{l}.*part_mm);
                    deltao_s{l} = delta{l}.*part_ms+delta_s{l}.*(-0.5*dsigmoid(sigmoid_v)/lambda.*(ratio_v.^3+ratio_v)+sigmoid_v ...
                        +0.5/sqrt(2*pi)*o0{l}.*exp_sq.*(ratio_v.^2./sqrt(o0s{l})+1./sqrt(o0s{l}))-2*h0{l}.*part_ms);

                    clear ratio_v sigmoid_v part_mm part_ms exp_sq;
                elseif my.use_tanh==1 % tanh
                    dsig = exp(-o0{l}.*kappa(o0s{l},1,0.25))./(1+exp(-o0{l}.*kappa(o0s{l},1,0.25))).^2;
                    part_mm = 2*dsig.*kappa(o0s{l},1,0.25);
                    part_ms = 2*dsig.*dkappa(o0s{l},1,0.25).*o0{l};
                    deltao{l} = ...
                        delta{l}.*part_mm+delta_s{l}.*(4*exp(-alphap*(o0{l}+betap).*kappa(o0s{l},alphap))./(1+exp(-alphap*(o0{l}+betap)...
                        .*kappa(o0s{l},alphap))).^2*alphap...
                        .*kappa(o0s{l},alphap)...
                        -4*dsig.*kappa(o0s{l},1,0.25)-2*h0{l}.*part_mm);
                    deltao_s{l} = ...
                        delta{l}.*part_ms+delta_s{l}.*(4*exp(-alphap*(o0{l}+betap).*kappa(o0s{l},alphap))./(1+exp(-alphap*(o0{l}+betap)...
                        .*kappa(o0s{l},alphap))).^2*alphap.*(o0{l}+betap).*dkappa(o0s{l},alphap)-4*dsig.*o0{l}.*dkappa(o0s{l},1,0.25)...
                        -2*h0{l}.*part_ms);

                    clear dsig part_mm part_ms;
                end
            end

            % handle dropout
            if M.dropout.use && l < n_layers && l > 1
                deltao{l} = deltao{l} .* h0mask{l};
                deltao_s{l} = deltao_s{l} .* h0mask{l};
            end

            % calculate gradients of parameters
            if l > 1
                biases_grad{l} = biases_grad{l} + mean(deltao{l}, 1);
                if my.positive_s==1
                    piases_s_grad{l} = piases_s_grad{l} + ...
                        mean(deltao_s{l}, 1).*exp(M.piases_s{l}')./...
                        (1+exp(M.piases_s{l}'));
                else
                    piases_s_grad{l} = piases_s_grad{l} + ...
                        mean(deltao_s{l}, 1);
                end
            end
            W_grad{l} = W_grad{l} + (h0{l}' * deltao{l+1}) / (size(v0, 1))...
                +2*(h0s{l}'*deltao_s{l+1}).*M.W{l}/size(v0,1);
            if my.positive_s==1
                M_s_grad{l} = M_s_grad{l}+(h0s{l}'*deltao_s{l+1}/size(v0,1)...
                    +(h0{l}.*h0{l})'*deltao_s{l+1}/size(v0,1))...
                    .*exp(M.M_s{l})./(1+exp(M.M_s{l}));
            else
                M_s_grad{l} = M_s_grad{l}+(h0s{l}'*deltao_s{l+1}/size(v0,1)...
                    +(h0{l}.*h0{l})'*deltao_s{l+1}/size(v0,1));
            end
        end

        clear h0mask;

        % learning rate
        % default: adadelta
        if M.adagrad.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                piases_s_grad_old{l} = (1 - momentum) * piases_s_grad{l} + momentum * piases_s_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                    M_s_grad_old{l} = (1 - momentum) * M_s_grad{l} + momentum* M_s_grad_old{l};
                end
            end

            for l = 1:n_layers
                if l < n_layers
                    M.adagrad.W{l} = M.adagrad.W{l} + W_grad_old{l}.^2;
                    M.adagrad.M_s{l} = M.adagrad.M_s{l} + M_s_grad_old{l}.^2;
                end

                M.adagrad.biases{l} = M.adagrad.biases{l} + biases_grad_old{l}.^2';
                M.adagrad.piases_s{l} = M.adagrad.piases_s{l} + piases_s_grad_old{l}.^2';
            end

            for l = 1:n_layers
                if step>my.fix_W_m || my.init_W_m==0
                    M.biases{l} = M.biases{l} - M.learning.lrate * (biases_grad_old{l}' + ...
                        weight_decay * M.biases{l}) ./ sqrt(M.adagrad.biases{l} + M.adagrad.epsilon);
                end
                M.piases_s{l} = M.piases_s{l} - M.learning.lrate * (piases_s_grad_old{l}' + ...
                    weight_decay_s * ...
                    (my.pi*(log(1+exp(M.piases_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.piases_s{l}))-my.reg_s2)).*...
                    exp(M.piases_s{l})./(1+exp(M.piases_s{l})) ) ./ sqrt(M.adagrad.piases_s{l} + M.adagrad.epsilon);
                if my.positive_s==1
                    M.biases_s{l} = log(1+exp(M.piases_s{l}));
                else
                    M.biases_s{l} = M.piases_s{l};
                end
                if l < n_layers
                    if step>my.fix_W_m || my.init_W_m==0
                        M.W{l} = M.W{l} - M.learning.lrate * (W_grad_old{l} + ...
                            weight_decay * M.W{l}) ./ sqrt(M.adagrad.W{l} + M.adagrad.epsilon);
                    end
                    M.M_s{l} = M.M_s{l} - M.learning.lrate * (M_s_grad_old{l} + ...
                        weight_decay_s * ...
                        (my.pi*(log(1+exp(M.M_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.M_s{l}))-my.reg_s2)).*...
                        exp(M.M_s{l})./(1+exp(M.M_s{l})) ) ./ sqrt(M.adagrad.M_s{l} + M.adagrad.epsilon);
                    if my.positive_s==1
                        M.W_s{l} = log(1+exp(M.M_s{l}));
                    else
                        M.W_s{l} = M.M_s{l};
                    end
                end
            end

        elseif M.adadelta.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                piases_s_grad_old{l} = (1 - momentum) * piases_s_grad{l} + momentum * piases_s_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                    M_s_grad_old{l} = (1 - momentum) * M_s_grad{l} + momentum * M_s_grad_old{l};
                end
            end

            if M.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = M.adadelta.momentum;
            end

            for l = 1:n_layers
                if l < n_layers
                    M.adadelta.gW{l} = adamom * M.adadelta.gW{l} + (1 - adamom) * W_grad_old{l}.^2;
                    M.adadelta.gM_s{l} = adamom * M.adadelta.gM_s{l} + (1 - adamom) * M_s_grad_old{l}.^2;
                end

                M.adadelta.gbiases{l} = adamom * M.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2';
                M.adadelta.gpiases_s{l} = adamom * M.adadelta.gpiases_s{l} + (1 - adamom) * piases_s_grad_old{l}.^2';
            end

            % When weight_decay_s is not 0, use product of Gaussian (PoG) prior.
            % Usually weight_decay_s = 0 when dropout is used.
            % For details see the paper at wanghao.in/paper/IJCAI13_CTRSR.pdf
            for l = 1:n_layers
                dbias = -(biases_grad_old{l}' + ...
                    weight_decay * M.biases{l}) .* (sqrt(M.adadelta.biases{l} + M.adadelta.epsilon) ./ ...
                    sqrt(M.adadelta.gbiases{l} + M.adadelta.epsilon));
                dpias_s = -(piases_s_grad_old{l}' + ...
                    weight_decay_s * ...
                    (my.pi*(log(1+exp(M.piases_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.piases_s{l}))-my.reg_s2)).*...
                    exp(M.piases_s{l})./(1+exp(M.piases_s{l})) ) .* (sqrt(M.adadelta.piases_s{l} + M.adadelta.epsilon) ./ ...
                    sqrt(M.adadelta.gpiases_s{l} + M.adadelta.epsilon));
                if step>my.fix_W_m || my.init_W_m==0
                    M.biases{l} = M.biases{l} + dbias;
                end
                M.piases_s{l} = M.piases_s{l} + dpias_s;
                if my.positive_s==1
                    M.biases_s{l} = log(1+exp(M.piases_s{l}));
                else
                    M.biases_s{l} = M.piases_s{l};
                end

                M.adadelta.biases{l} = adamom * M.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                M.adadelta.piases_s{l} = adamom * M.adadelta.piases_s{l} + (1 - adamom) * dpias_s.^2;
                clear dbias dpias_s;

                if l < n_layers
                    dW = -(W_grad_old{l} + ...
                        weight_decay * M.W{l}) .* (sqrt(M.adadelta.W{l} + M.adadelta.epsilon) ./ ...
                        sqrt(M.adadelta.gW{l} + M.adadelta.epsilon));
                    dM_s = -(M_s_grad_old{l} + ...
                        weight_decay_s * ...
                        (my.pi*(log(1+exp(M.M_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.M_s{l}))-my.reg_s2)).*...
                        exp(M.M_s{l})./(1+exp(M.M_s{l})) ) .* (sqrt(M.adadelta.M_s{l} + M.adadelta.epsilon) ./ ...
                        sqrt(M.adadelta.gM_s{l} + M.adadelta.epsilon));
                    if step>my.fix_W_m || my.init_W_m==0
                        M.W{l} = M.W{l} + dW;
                    end
                    M.M_s{l} = M.M_s{l} + dM_s;
                    if my.positive_s==1
                        M.W_s{l} = log(1+exp(M.M_s{l}));
                    else
                        M.W_s{l} = M.M_s{l};
                    end

                    M.adadelta.W{l} = adamom * M.adadelta.W{l} + (1 - adamom) * dW.^2;
                    M.adadelta.M_s{l} = adamom * M.adadelta.M_s{l} + (1 - adamom) * dM_s.^2;

                    clear dW dM_s;
                end

            end
        else
            if M.learning.lrate_anneal > 0 && (step >= M.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if M.learning.lrate0 > 0
                    actual_lrate = M.learning.lrate / (1 + M.iteration.n_updates / M.learning.lrate0);
                else
                    actual_lrate = M.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            M.signals.lrates = [M.signals.lrates actual_lrate];

            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                piases_s_grad_old{l} = (1 - momentum) * piases_s_grad{l} + momentum * piases_s_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                    M_s_grad_old{l} = (1 - momentum) * M_s_grad{l} + momentum * M_s_grad_old{l};
                end
            end

            for l = 1:n_layers
                if step>my.fix_W_m || my.init_W_m==0
                    M.biases{l} = M.biases{l} - actual_lrate * (biases_grad_old{l}' + weight_decay * M.biases{l});
                end
                M.piases_s{l} = M.piases_s{l} - actual_lrate * (piases_s_grad_old{l}' + ...
                    weight_decay_s * ...
                    (my.pi*(log(1+exp(M.piases_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.piases_s{l}))-my.reg_s2)).*...
                    exp(M.piases_s{l})./(1+exp(M.piases_s{l})) );
                if my.positive_s==1
                    M.biases_s{l} = log(1+exp(M.piases_s{l}));
                else
                    M.biases_s{l} = M.piases_s{l};
                end
                if l < n_layers
                    if step>my.fix_W_m || my.init_W_m==0
                        M.W{l} = M.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * M.W{l});
                    end
                    M.M_s{l} = M.M_s{l} - actual_lrate * (M_s_grad_old{l} + ...
                        weight_decay_s * ...
                        (my.pi*(log(1+exp(M.M_s{l}))-my.reg_s1)+(1-my.pi)*(log(1+exp(M.M_s{l}))-my.reg_s2)).*...
                        exp(M.M_s{l})./(1+exp(M.M_s{l})) );
                    if my.positive_s==1
                        M.W_s{l} = log(1+exp(M.M_s{l}));
                    else
                        M.W_s{l} = M.M_s{l};
                    end
                end
            end
        end

        if M.verbose == 1
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid, '.');
            fclose(myfid);
            fprintf(2, '.');
        end

        if use_gpu > 0
            clear v0 v0s h0d h0e v0_clean vr hr deltae deltad a_rand
        end

        if early_stop
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            v0valid = valid_patches(rndidx(1:round(n_valid * valid_portion)),:);
            if use_gpu > 0
                v0valid = gpuArray(single(v0valid));
            end

            if M.output.binary
                vr = mlp_classify(M, v0valid, [], 1);
            else
                vr = mlp_classify(M, v0valid);
            end
            if use_gpu > 0
                vr = gather(vr);
            end

            if M.output.binary
                xt = valid_targets(rndidx(1:round(n_valid * valid_portion)), :);
                yt = vr;
                [mp, mi] = max(gather(yt), [], 2);
                [tp, ti] = max(gather(xt), [], 2);

                n_correct = sum(mi == ti);
                rerr = 1 - n_correct/(round(n_valid * valid_portion));
            else
                rerr = mean(sum((valid_targets(rndidx(1:round(n_valid * valid_portion), :)) - vr).^2,2));
            end
            if use_gpu > 0
                rerr = gather(rerr);
            end

            M.signals.valid_errors = [M.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;

                if step > M.valid_min_epochs && (1.1 * valid_best_err) < valid_err 
                    myfid = fopen(sprintf('%s.log',my.save),'a');
                    fprintf(myfid, 'Early-stop! %f, %f\n', valid_err, valid_best_err);
                    fclose(myfid);
                    fprintf(2, 'Early-stop! %f, %f\n', valid_err, valid_best_err);
                    stopping = 1;
                    break;
                end

                if valid_err < valid_best_err
                    valid_best_err = valid_err;
                end
            end
        else
            if M.stop.criterion > 0
                if M.stop.criterion == 1
                    if min_recon_error > M.signals.recon_errors(end)
                        min_recon_error = M.signals.recon_errors(end);
                        min_recon_error_update_idx = M.iteration.n_updates;
                    else
                        if M.iteration.n_updates > min_recon_error_update_idx + M.stop.recon_error.tolerate_count 
                            myfid = fopen(sprintf('%s.log',my.save),'a');
                            fprintf(myfid, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                M.signals.recon_errors(end), min_recon_error);
                            fclose(myfid);
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                M.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', M.stop.criterion);
                end
            end
        end

        if length(M.hook.per_update) > 1
            err = M.hook.per_update{1}(M, M.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if M.debug.do_display == 1 && mod(M.iteration.n_updates, M.debug.display_interval) == 0
            M.debug.display_function (M.debug.display_fid, M, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end

    if use_gpu > 0
        % pull
        for l = 1:n_layers
            if l < n_layers
                M.W{l} = gather(M.W{l});
                M.M_s{l} = gather(M.M_s{l});
                M.W_s{l} = gather(M.W_s{l});
            end
            M.biases{l} = gather(M.biases{l});
            M.piases_s{l} = gather(M.piases_s{l});
            M.biases_s{l} = gather(M.biases_s{l});
        end

        if M.adagrad.use
            for l = 1:n_layers
                if l < n_layers
                    M.adagrad.W{l} = gather(M.adagrad.W{l});
                    M.adagrad.M_s{l} = gather(M.adagrad.M_s{l});
                end
                M.adagrad.biases{l} = gather(M.adagrad.biases{l});
                M.adagrad.piases_s{l} = gather(M.adagrad.piases_s{l});
            end
        elseif M.adadelta.use
            for l = 1:n_layers
                if l < n_layers
                    M.adadelta.W{l} = gather(M.adadelta.W{l});
                    M.adadelta.gW{l} = gather(M.adadelta.gW{l});
                    M.adadelta.M_s{l} = gather(M.adadelta.M_s{l});
                    M.adadelta.gM_s{l} = gather(M.adadelta.gM_s{l});
                end
                M.adadelta.biases{l} = gather(M.adadelta.biases{l});
                M.adadelta.gbiases{l} = gather(M.adadelta.gbiases{l});
                M.adadelta.piases_s{l} = gather(M.adadelta.piases_s{l});
                M.adadelta.gpiases_s{l} = gather(M.adadelta.gpiases_s{l});
            end
        end
    end

    if length(M.hook.per_epoch) > 1
        err = M.hook.per_epoch{1}(M, M.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if M.verbose == 1
        myfid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(myfid, '\n');
        fclose(myfid);
        fprintf(2, '\n');
    end
        

    if my.regression_target~=0
        myfid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(myfid, 'Epoch %d/%d - rerr/LL/time: %f/%1.1f/%1.1f\n', step, n_epochs, ...
            mean(M.signals.recon_errors(end-n_minibatches+1:end)), ...
            mean(M.signals.train_LL(end-n_minibatches+1:end)), toc);
        fclose(myfid);
        fprintf(2, 'Epoch %d/%d - rerr/LL/time: %f/%1.1f/%1.1f\n', step, n_epochs, ...
            mean(M.signals.recon_errors(end-n_minibatches+1:end)), ...
            mean(M.signals.train_LL(end-n_minibatches+1:end)), toc);
    end
    myfid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(myfid,'Epoch %d/%d - rerr/time: %f/%1.1f\n',...
        step,n_epochs,rerr,toc);
    fclose(myfid);
    fprintf(2,'Epoch %d/%d - rerr/time: %f/%1.1f\n',...
        step,n_epochs,rerr,toc);

    if mod(step,my.test_interval)==0
        if my.regression_target==0
            [pred] = mlp_classify_bayes (M, X_test, X_test_labels, my);
            n_correct = sum(X_test_labels == pred);
            
            fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
            fclose(myfid);
        else
            [pred pred_s] = mlp_classify_bayes (M, X_test, X_test_labels, my);
            RMSE = sqrt(mean(sum((pred-X_test_labels).^2,2)));
            fprintf(2,'Test RMSE is: %1.4f\n',RMSE);
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid,'Test RMSE is: %1.4f\n',RMSE);
            % log-likelihood
            LL = mean(-0.5./pred_s.*(pred-X_test_labels).^2-0.5*log(2*pi*pred_s));
            fprintf(2,'Test LL is: %1.4f\n',LL);
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid,'Test LL is: %1.4f\n',LL);
            fclose(myfid);
        end
        % tmp save
        if exist(sprintf('%s-%d.mat',my.save,step-my.test_interval),'file')==2
            system(sprintf('rm %s-%d.mat',my.save,step-my.test_interval));
        end
        if step~=n_epochs
            save(sprintf('%s-%d.mat',my.save,step),'M','my');
        end
    end
end

if use_gpu > 0
    % pull
    for l = 1:n_layers
        if l < n_layers
            M.W{l} = gather(M.W{l});
            M.M_s{l} = gather(M.M_s{l});
            M.W_s{l} = gather(M.W_s{l});
        end
        M.biases{l} = gather(M.biases{l});
        M.piases_s{l} = gather(M.piases_s{l});
        M.biases_s{l} = gather(M.biases_s{l});
    end

    if M.adagrad.use
        for l = 1:n_layers
            if l < n_layers
                M.adagrad.W{l} = gather(M.adagrad.W{l});
                M.adagrad.M_s{l} = gather(M.adagrad.M_s{l});
            end
            M.adagrad.biases{l} = gather(M.adagrad.biases{l});
            M.adagrad.piases_s{l} = gather(M.adagrad.piases_s{l});
        end
    elseif M.adadelta.use
        for l = 1:n_layers
            if l < n_layers
                M.adadelta.W{l} = gather(M.adadelta.W{l});
                M.adadelta.gW{l} = gather(M.adadelta.gW{l});
                M.adadelta.M_s{l} = gather(M.adadelta.M_s{l});
                M.adadelta.gM_s{l} = gather(M.adadelta.gM_s{l});
            end
            M.adadelta.biases{l} = gather(M.adadelta.biases{l});
            M.adadelta.gbiases{l} = gather(M.adadelta.gbiases{l});
            M.adadelta.piases_s{l} = gather(M.adadelta.piases_s{l});
            M.adadelta.gpiases_s{l} = gather(M.adadelta.gpiases_s{l});
        end
    end
end


