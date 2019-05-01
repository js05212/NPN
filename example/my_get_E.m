function my_get_E
% load data and get error of each item
addpath('..');
p = '79';
load(sprintf('lapnn%s.mat',p));
X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
targets = loadMNISTLabels('t10k-labels.idx1-ubyte');
%X_test = loadMNISTImages('train-images.idx3-ubyte')';
%targets = loadMNISTLabels('train-labels.idx1-ubyte');
C=unique(targets);
t = bsxfun(@eq,repmat(targets,1,length(C)),repmat(C',length(targets),1));
[m s] = lapnn_get_m_s(M,X_test,t,my);
E = sum(t.*log(max(m,1e-16))+(1-t).*log(max(1-m,1e-16)),2);
s_sum = sum(s,2);
[v i] = max(m,[],2);
pred = (i==targets+1);
fprintf('%d rights\n',sum(pred));

mkdir(sprintf('lapnn%s',p));
cd(sprintf('lapnn%s',p));
dlmwrite(sprintf('E.%s',p),E,'delimiter',' ');
dlmwrite(sprintf('s_sum.%s',p),s_sum,'delimiter',' ');
dlmwrite(sprintf('pred.%s',p),pred,'delimiter',' ');
