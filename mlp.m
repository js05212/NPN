% mlp - training an MLP (stochastic backprop)
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
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
function [M] = mlp(M, patches, targets, X_test, X_test_labels, my, valid_patches, valid_targets, ...
    valid_portion, use_cvp);

if nargin < 7 % modified by hog from 3 to 7
    early_stop = 0;
    valid_patches = [];
    valid_targets = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end

if nargin < 9
    use_cvp = 0; % modified by hog to 0
end

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
    % new_targets = strcmp(repmat(targets,1,length(C)),repmat(C',length(targets),1));
    new_targets = bsxfun(@eq,repmat(targets,1,length(C)),repmat(C',length(targets),1));
%    n_classes = max(targets(:));
%    new_targets = zeros(n_samples, n_classes);

%    for ti = 1:n_samples
%        new_targets(ti, targets(ti)) = 1; % well?
%    end
    
    targets = new_targets;
end

if size(valid_targets, 2) == 1 && M.output.binary
    % populate the target labels
    % more efficient way to populate the target labels thanks to vikrantt
    C=unique(valid_targets);
    % new_targets = strcmp(repmat(targets,1,length(C)),repmat(C',length(targets),1));
    new_targets = bsxfun(@eq,repmat(valid_targets,1,length(C)),repmat(C',length(valid_targets),1));
%    n_classes = max(valid_targets(:));
%    n_valid_samples = size(valid_targets, 1);
%    new_targets = zeros(n_valid_samples, n_classes);

%    for ti = 1:n_valid_samples
%        new_targets(ti, valid_targets(ti)) = 1; % well?
%    end
    
    valid_targets = new_targets;
end

n_epochs = M.iteration.n_epochs;

momentum = M.learning.momentum;
weight_decay = M.learning.weight_decay;

biases_grad = cell(n_layers, 1);
W_grad = cell(n_layers, 1);
biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad{l} = zeros(size(M.biases{l}))';
    if l < n_layers
        W_grad{l} = zeros(size(M.W{l}));
    end
    biases_grad_old{l} = zeros(size(M.biases{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(M.W{l}));
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = M.do_normalize;
do_normalize_std = M.do_normalize_std;

if M.data.binary == 0
    if do_normalize == 1
        % make it zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
    end

    if do_normalize_std ==1
        % make it unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
    end
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
            end
            M.biases{l} = gpuArray(single(M.biases{l}));
        end

        if M.adagrad.use 
            for l = 1:n_layers
                if l < n_layers 
                    M.adagrad.W{l} = gpuArray(single(M.adagrad.W{l}));
                end
                M.adagrad.biases{l} = gpuArray(single(M.adagrad.biases{l}));
            end
        elseif M.adadelta.use
            for l = 1:n_layers
                if l < n_layers 
                    M.adadelta.gW{l} = gpuArray(single(M.adadelta.gW{l}));
                    M.adadelta.W{l} = gpuArray(single(M.adadelta.W{l}));
                end
                M.adadelta.gbiases{l} = gpuArray(single(M.adadelta.gbiases{l}));
                M.adadelta.biases{l} = gpuArray(single(M.adadelta.biases{l}));
            end
        end
    end

    for mb=1:n_minibatches
        M.iteration.n_updates = M.iteration.n_updates + 1;

        if use_cvp
            v0 = patches(cvp == mb, :);
            t0 = targets(cvp == mb, :);
        else
            mb_start = (mb - 1) * minibatch_sz + 1;
            mb_end = min(mb * minibatch_sz, n_samples);

            % p_0
            v0 = patches(mb_start:mb_end, :);
            t0 = targets(mb_start:mb_end, :);
        end
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        %if M.data.binary == 0 && M.noise.level > 0
        if M.noise.level > 0
            v0 = v0 + M.noise.level * gpuArray(randn(size(v0)));
        end

        if M.noise.drop > 0
            mask = binornd(1, 1 - M.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end

        h0 = cell(n_layers, 1);
        h0mask = cell(n_layers, 1);
        h0{1} = v0;

        for l = 2:n_layers
            h0{l} = bsxfun(@plus, h0{l-1} * M.W{l-1}, M.biases{l}');

            if l < n_layers
                h0{l} = sigmoid(h0{l}, M.hidden.use_tanh);
            end

            if M.dropout.use && l < n_layers
                h0mask{l} = single(bsxfun(@minus, rand(size(h0{l})), ...
                    M.dropout.probs{l}') > 0); % bug fixed by hog
                h0{l} = h0mask{l} .* h0{l};
            end

            if l == n_layers && M.output.binary
                h0{l} = sigmoid(h0{l});
            end
        end

        % reset gradients
        for l = 1:n_layers
            biases_grad{l} = 0 * biases_grad{l};
            if l < n_layers
                W_grad{l} = 0 * W_grad{l};
            end
        end

        % backprop
        delta = cell(n_layers, 1);
        delta{end} = h0{end} - t0;

        if M.output.binary
            if use_cvp
                xt = targets(cvp == mb, :);
            else
                xt = targets(mb_start:mb_end, :);
            end
            rerr = -mean(sum(xt .* log(max(h0{end}, 1e-16)) + (1 - xt) .* log(max(1 - h0{end}, 1e-16)), 2));
        else
            rerr = mean(sum(delta{end}.^2,2));
        end
        if use_gpu > 0
            rerr = gather(rerr);
        end
        M.signals.recon_errors = [M.signals.recon_errors rerr];

        biases_grad{end} = mean(delta{end}, 1);

        for l = n_layers-1:-1:1
            delta{l} = delta{l+1} * M.W{l}';
            if l == 1 && M.data.binary
                delta{l} = delta{l} .* dsigmoid(h0{l});
            end
            if l > 1
                delta{l} = delta{l} .* dsigmoid(h0{l}, M.hidden.use_tanh);
            end

            if M.dropout.use && l < n_layers && l > 1
                delta{l} = delta{l} .* h0mask{l};
            end

            if l > 1
                biases_grad{l} = biases_grad{l} + mean(delta{l}, 1);
            end
            W_grad{l} = W_grad{l} + (h0{l}' * delta{l+1}) / (size(v0, 1));
        end

        clear h0mask;

        % learning rate
        if M.adagrad.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                if l < n_layers
                    M.adagrad.W{l} = M.adagrad.W{l} + W_grad_old{l}.^2;
                end

                M.adagrad.biases{l} = M.adagrad.biases{l} + biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                M.biases{l} = M.biases{l} - M.learning.lrate * (biases_grad_old{l}' + ...
                    weight_decay * M.biases{l}) ./ sqrt(M.adagrad.biases{l} + M.adagrad.epsilon);
                if l < n_layers
                    M.W{l} = M.W{l} - M.learning.lrate * (W_grad_old{l} + ...
                        weight_decay * M.W{l}) ./ sqrt(M.adagrad.W{l} + M.adagrad.epsilon);
                end
            end

        elseif M.adadelta.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
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
                end

                M.adadelta.gbiases{l} = adamom * M.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                dbias = -(biases_grad_old{l}' + ...
                    weight_decay * M.biases{l}) .* (sqrt(M.adadelta.biases{l} + M.adadelta.epsilon) ./ ...
                    sqrt(M.adadelta.gbiases{l} + M.adadelta.epsilon));
                M.biases{l} = M.biases{l} + dbias;

                M.adadelta.biases{l} = adamom * M.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                if l < n_layers
                    dW = -(W_grad_old{l} + ...
                        weight_decay * M.W{l}) .* (sqrt(M.adadelta.W{l} + M.adadelta.epsilon) ./ ...
                        sqrt(M.adadelta.gW{l} + M.adadelta.epsilon));
                    M.W{l} = M.W{l} + dW;

                    M.adadelta.W{l} = adamom * M.adadelta.W{l} + (1 - adamom) * dW.^2;

                    clear dW;
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
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                M.biases{l} = M.biases{l} - actual_lrate * (biases_grad_old{l}' + weight_decay * M.biases{l});
                if l < n_layers
                    M.W{l} = M.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * M.W{l});
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
            clear v0 h0d h0e v0_clean vr hr deltae deltad 
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
            end
            M.biases{l} = gather(M.biases{l});
        end

        if M.adagrad.use
            for l = 1:n_layers
                if l < n_layers
                    M.adagrad.W{l} = gather(M.adagrad.W{l});
                end
                M.adagrad.biases{l} = gather(M.adagrad.biases{l});
            end
        elseif M.adadelta.use
            for l = 1:n_layers
                if l < n_layers
                    M.adadelta.W{l} = gather(M.adadelta.W{l});
                    M.adadelta.gW{l} = gather(M.adadelta.gW{l});
                end
                M.adadelta.biases{l} = gather(M.adadelta.biases{l});
                M.adadelta.gbiases{l} = gather(M.adadelta.gbiases{l});
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
        
    myfid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(myfid, 'Epoch %d/%d - rerr/time: %f/%1.1f\n', step, n_epochs, ...
        M.signals.recon_errors(end), toc);
    fclose(myfid);
    fprintf(2, 'Epoch %d/%d - rerr/time: %f/%1.1f\n', step, n_epochs, ...
        M.signals.recon_errors(end), toc);

    if mod(step,my.test_interval)==0
        if my.regression_target==0
            [pred] = mlp_classify (M, X_test, X_test_labels, my);
            n_correct = sum(X_test_labels == pred);
            
            fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
            fclose(myfid);
        else
            [pred] = mlp_classify (M, X_test, X_test_labels, my);
            MSE = mean(sum((pred-X_test_labels).^2,2));
            fprintf(2,'Test MSE is: %1.1f\n',MSE);
            myfid = fopen(sprintf('%s.log',my.save),'a');
            fprintf(myfid,'Test MSE is: %1.1f\n',MSE);
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
        end
        M.biases{l} = gather(M.biases{l});
    end

    if M.adagrad.use
        for l = 1:n_layers
            if l < n_layers
                M.adagrad.W{l} = gather(M.adagrad.W{l});
            end
            M.adagrad.biases{l} = gather(M.adagrad.biases{l});
        end
    elseif M.adadelta.use
        for l = 1:n_layers
            if l < n_layers
                M.adadelta.W{l} = gather(M.adadelta.W{l});
                M.adadelta.gW{l} = gather(M.adadelta.gW{l});
            end
            M.adadelta.biases{l} = gather(M.adadelta.biases{l});
            M.adadelta.gbiases{l} = gather(M.adadelta.gbiases{l});
        end
    end
end


