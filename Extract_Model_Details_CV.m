function [weights, decisionValues] = Extract_Model_Details_CV(dependentVars, independentVars)
    % extract each cross validation' weight and decision value of svm model
    % weights: feature dimesions * sample size; decisionValues: sample size
    decisionValues = zeros(numel(dependentVars),1); % initial decision values
    weights = zeros(size(independentVars, 2),numel(dependentVars)); % 存放features*iters的权重
        for i = 1:numel(dependentVars)
            % 分配训练集和验证集
            test_set = independentVars(i,:);
            train_set = independentVars;
            train_set(i,:) = [];
            test_label = dependentVars(i,1);
            train_label = dependentVars;
            train_label(i,:) = [];
            model = svmtrain(train_label,train_set, '-s 3 -q -t 0');
            weights(:,i) = Cal_svm_paras(model);
            [~, ~, dec_value] = svmpredict(test_label, test_set,model);
            decisionValues(i) = dec_value;
        end
end
