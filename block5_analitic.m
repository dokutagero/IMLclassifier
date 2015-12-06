data = load('diabetes.mat');
train_size = round(size(data.x,2)*i/times);
train_data = data.x(:,1:train_size);
train_target = data.y(1:train_size);
test_data = data.x(:,train_size+1:end);
test_target = data.y(train_size+1:end);

N(i) = train_size;

%Process data
[D2_train,c1_means,c2_means] = preprocessData2(train_data,train_target);

[meanColumns, stdevColumns, D2_train] = normalization(D2_train');
D2_train = D2_train';

% Linear regression
[ w_train, y_predicted]=getLinearRegressionOutput( D2_train, train_target );

%Process test data
D2_test = test_data;

for j=1:size(test_data,1)
    attr = test_data(j,:);
    attr_no_nan = ~isnan(test_data(j,:));
    c = test_target .* attr_no_nan';
    c_nan = test_target .* ~attr_no_nan';
    
    D2_test(j,c_nan == 1) = c1_means(j);
    D2_test(j,c_nan == -1) = c2_means(j);
end

[meanColumns, stdevColumns, D2_test] = normalization(D2_test',meanColumns, stdevColumns);
D2_test = D2_test';


% Measure errors
[ y_classified_train ] = linearClassifier(D2_train,w_train);
train_errors = sum(y_classified_train ~= train_target);
train_err_rate = (train_errors/train_size) *100;

[ y_classified_test ] = linearClassifier(D2_test,w_train);
test_errors = sum(y_classified_test ~= test_target);
test_err_rate = (test_errors/length(test_target)) *100;