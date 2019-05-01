% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');
clear

my.gpu_id = 2;
my.save = 'lapnn327';
my.n_hidden = 400;
my.dropout = 0.4;
my.use_tanh = 0;
my.is_debug = 0;
my.is_bayes = 1;
my.subtract_mean = 0;


my.do_train = 0;
my.folder = 2; % 1->196, 2->784, 3->toy regr'
my.print_wrongs = 0;
my.print_rights = 0;
my.use_second = 0;
my.thre.first = 1.9999999;
my.thre.second = 0.5;
my.thre.var_value1 = 0.00;
my.thre.var_value2 = 0;
my.thre.var_gap = 0.04;
my.thre.mean_gap = 0.16;

my.mc_dropout = 0;
rand('seed',11112);

my

gpuDevice(my.gpu_id);

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
    load 'toy_1d_regression.mat';
    my.minibatch = 128;
    my.epsilon = 1e-5;
end

if my.is_debug==1 && my.folder~=3
    X = X(1:500,:);
    X_labels = X_labels(1:500,:);
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
    layers = [size(X,2), 100, 1];
    n_layers = length(layers);
end

use_tanh = my.use_tanh;

M.output.binary = blayers(end);
M.hidden.use_tanh = use_tanh;

M.valid_min_epochs = 10;
if my.dropout~=0
    M.dropout.use = 1;
else
    M.dropout.use = 0;
end

M.hook.per_epoch = {@save_intermediate, {'mlp_mnist.mat'}};

% added by hog
for l = 1:n_layers
    M.dropout.probs{l} = my.dropout*ones(layers(l),1);
end

tic;

load(my.save);
my.m_gaps = linspace(0.0,0.2,11);
my.v_gaps = linspace(0,0.18,39);
my.m_val = linspace(0.2,0.9,12);
my.acc = zeros(length(my.m_gaps),length(my.v_gaps),length(my.m_val));

[pred] = cv_mlp_classify_bayes (M, X_test, X_test_labels, my);
