function process_boston
m = dlmread('housing.data');
m(:,1:13) = bsxfun(@minus,m(:,1:13),min(m(:,1:13)));
m(:,1:13) = bsxfun(@rdivide,m(:,1:13),max(m(:,1:13)));
X = m(1:456,1:13);
X_labels = m(1:456,14);
X_test = m(457:506,1:13);
X_test_labels = m(457:506,14);
%save('boston_housing','X','X_labels','X_test','X_test_labels');
save('boston_housing_nor','X','X_labels','X_test','X_test_labels');
