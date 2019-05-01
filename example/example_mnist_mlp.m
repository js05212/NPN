% add the path of RBM code
addpath('..');
clear

my.gpu_id =1;
my.save = 'lapnn620';
my.with_kappa = 0;
rand('seed',11112);
my.denoising_type = 5; % 1->mask2gauss 2->mixed_mask 3->gauss 4->mask 5->bn
my.denoising = 0.1; % = noise.drop
my.dropout = 0.2;
my.weight_decay_s = 0e-4;
my.n_hidden = 400;
my.output_s = 0e2; % reg' the output variance
my.mutual_info = 0; % 1-> min' I(X;Y) -1-> max' I(X;Y)
my.gaussian_s = 1; % the target is gaussian
my.is_debug = 2; % 0->full 1->500 2->100 3->2e3 4->1e4 5->com 6->c1e4 7->c6e4
my.pi = 0.5;
my.nlogs1 = 1; % 0/1/2
my.nlogs2 = 7; % 6/7/8
n_epochs = 2000;
my.from = 1;
my.weight_decay = 0e-3;
my.init_s = 0.0; % 0->all zeros
my.regression_target = 0; % 1->Gaussian 2->L2
my.adadelta = 1;
my.is_bayes = 1;
my.use_tanh = 0;
my.minibatch = 128;
% my.is_debug: con'd 8->c5e2_for_MAP 9->c1e4_for_MAP

my.denoising_gauss = 0.0;
my.mc_dropout = 0;
my.mc_weight = 0;
my.do_train = 1;
my.epsilon = 1e-8;
my.mixed_dropout = 0;
my.subtract_mean = 0;
my.deep_s = 0e-2;
my.folder = 2; % 1->196, 2->784, 3->regr'
my.print_wrongs = 0;
my.print_rights = 0;
my.use_second = 0;
my.thre.first = 1.9999999;
my.thre.second = 0.5;
my.thre.var_value1 = 0.00;
my.thre.var_value2 = 0;
my.thre.var_gap = 0.04;
my.thre.mean_gap = 0.16;
if my.use_tanh~=0
    my.with_kappa = 1;
end

my.reg_s = my.pi*(1/exp(my.nlogs1))^2+(1-my.pi)*(1/exp(my.nlogs2))^2;
my.reg_s1 = 1/exp(my.nlogs1)^2;
my.reg_s2 = 1/exp(my.nlogs2)^2;
my.test_interval = 50;
n_epochs_pre = 100;
my.init_W_m = 0; % mat file id to init' W_m
my.fix_W_m = 0; % #epochs to fix W_m
my.opposite_s = 0.0; % make variance of wrong labels large

my

my.positive_s = 1;
gpuDevice(my.gpu_id);
%gpuDevice;
my.early_stop_thre = 2;

system(sprintf('cp example_mnist_mlp.m used_m/example_mnist_mlp.m.%s.%s',...
    my.save,strrep(datestr(now),' ','_')));

% load MNIST
if my.folder==1
    load 'mnist_14x14.mat';
    my.regression_target = 0; % 1->Gaussian 2->L2
elseif my.folder==2
    X = loadMNISTImages('train-images.idx3-ubyte')';
    X_labels = loadMNISTLabels('train-labels.idx1-ubyte');
    X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
    X_test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    my.regression_target = 0; % 1->Gaussian 2->L2
elseif my.folder==3
    % toy regression task
    %load 'toy_1d_regression.mat';
    %my.minibatch = 128;
    %my.epsilon = 1e-5;

    % boston housing regression task
    load 'boston_housing_nor.mat';
    my.epsilon = 1e-7;
end

if my.is_debug==1 && my.folder~=3
    X = X(1:500,:);
    X_labels = X_labels(1:500,:);
elseif my.is_debug==2 && my.folder~=3
    my.minibatch = 32;
    X = X(1:100,:);
    X_labels = X_labels(1:100,:);
elseif my.is_debug==3 && my.folder~=3
    X = X(1:2000,:);
    X_labels = X_labels(1:2000,:);
elseif my.is_debug==4 && my.folder~=3
    X = X(1:10000,:);
    X_labels = X_labels(1:10000,:);
elseif my.is_debug==5 && my.folder~=3
    load 'mnist_com_1_50.mat';
    X = X_m_com;
    X_labels = X_labels_com;
    X_s = X_s_com;
elseif my.is_debug==6 && my.folder~=3
    load 'mnist_com_1_1000.mat';
    X = X_m_com;
    X_labels = X_labels_com;
    X_s = X_s_com;
elseif my.is_debug==7 && my.folder~=3
    load 'mnist_com_1_6000.mat';
    X = X_m_com;
    X_labels = X_labels_com;
    X_s = X_s_com;
elseif my.is_debug==8 && my.folder~=3
    load 'mnist_com_1_50.mat';
    X = X_m_com;
    X_labels = X_labels_com;
elseif my.is_debug==9 && my.folder~=3
    load 'mnist_com_1_1000.mat';
    X = X_m_com;
    X_labels = X_labels_com;
end


% shuffle the training data
X_labels = X_labels + 1;
X_test_labels = X_test_labels + 1;

perm_idx = 1:size(X,1);%randperm (size(X,1));

n_all = size(X, 1);
n_train = n_all;%ceil(n_all * 3 / 4);
n_valid = 0;%floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), my.n_hidden, my.n_hidden, 10];
n_layers = length(layers);
blayers = [1, 1, 1, 1];

if my.folder==3
    blayers = [1, 1, 0];
    layers = [size(X,2), 50, 1];
    n_layers = length(layers);
end

use_tanh = my.use_tanh;

if my.is_bayes==1
    M = default_mlp_bayes (layers);
else
    M = default_mlp (layers);
end

M.output.binary = blayers(end);
M.hidden.use_tanh = use_tanh;

M.valid_min_epochs = 10;
if my.dropout~=0
    M.dropout.use = 1;
else
    M.dropout.use = 0;
end

M.hook.per_epoch = {@save_intermediate, {'mlp_mnist.mat'}};

M.learning.lrate = 1e-5; % original: 1e-3
M.learning.lrate0 = 5000; % original: 5000
M.learning.minibatch_sz = my.minibatch;

M.learning.weight_decay_s = my.weight_decay_s;
M.learning.weight_decay = my.weight_decay;

M.adadelta.use = my.adadelta;
M.adadelta.epsilon = my.epsilon; % original: 1e-8
M.adadelta.momentum = 0.99; % original: 0.99

M.noise.drop = my.denoising;
M.noise.level = my.denoising_gauss;

M.iteration.n_epochs = n_epochs;

% added by hog
for l = 1:n_layers
    M.dropout.probs{l} = my.dropout*ones(layers(l),1);
end

fprintf(1, 'Training MLP\n');
myfid = fopen(sprintf('%s.log',my.save),'a');
fprintf(myfid, 'Training MLP\n');
fclose(myfid);
tic;
if my.do_train==1
    if my.from~=1
        oldmy = my;
        load(my.save);
        my = oldmy;
        M.iteration.n_epochs = n_epochs;
    end
    % handle init' W_m
    if my.init_W_m~=0
        oldM = M;
        oldmy = my;
        load(sprintf('lapnn%d',my.init_W_m));
        for i = 1:length(size(oldM.structure.layers))
            oldM.W{i} = M.W{i};
            oldM.biases{i} = M.biases{i};
        end
        M = oldM;
        my = oldmy;
    end
    if my.is_bayes==1
        if my.is_debug>=5 && my.is_debug<=7
            M = mlp_bayes (M, X, X_labels, X_test, X_test_labels, my, X_s);%, X_valid, X_valid_labels, 0.1, 1);
        else
            M = mlp_bayes (M, X, X_labels, X_test, X_test_labels, my);%, X_valid, X_valid_labels, 0.1, 1);
        end
    else
        M = mlp (M, X, X_labels, X_test, X_test_labels, my);%, X_valid, X_valid_labels, 0.1, 1);
    end
    save(my.save,'M','my');
else
    oldmy = my;
    load(my.save);
    my = oldmy;
end
fprintf(1, 'Training is done after %f seconds\n', toc);
myfid = fopen(sprintf('%s.log',my.save),'a');
fprintf(myfid, 'Training is done after %f seconds\n', toc);
fclose(myfid);

if my.is_bayes==1
    if my.mc_dropout~=0 && my.mc_weight==0
        [pred pred_s] = mc_mlp_classify_bayes (M, X_test, X_test_labels, my);
    elseif my.mc_dropout==0 && my.mc_weight==0
        [pred pred_s] = mlp_classify_bayes (M, X_test, X_test_labels, my);
    elseif my.mc_weight~=0
        [pred pred_s] = mcw_mlp_classify_bayes (M, X_test, X_test_labels, my);
    end
else
    if my.mc_dropout~=0
        [pred] = mc_mlp_classify (M, X_test, X_test_labels, my);
    else
        [pred] = mlp_classify (M, X_test, X_test_labels, my);
    end
end

if my.regression_target==0
    n_correct = sum(X_test_labels == pred);
    fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
    myfid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(myfid, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
    fclose(myfid);
else
    % RMSE
    RMSE = sqrt(mean(sum((pred-X_test_labels).^2,2)));
    fprintf(2,'Test RMSE is: %1.4f\n',RMSE);
    myfid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(myfid,'Test RMSE is: %1.4f\n',RMSE);
    if my.is_bayes==1
        % log-likelihood
        LL = mean(-0.5./pred_s.*(pred-X_test_labels).^2-0.5*log(2*pi*pred_s));
        fprintf(2,'Test LL is: %1.4f\n',LL);
        myfid = fopen(sprintf('%s.log',my.save),'a');
        fprintf(myfid,'Test LL is: %1.4f\n',LL);
    end
    fclose(myfid);
end




