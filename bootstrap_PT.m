function [Dist_max,Dist_min, Dist_h] = bootstrap_PT(label,insts,perm_times)
% using maximum statistics implicitly accounts for multiple comparisons (Nichols and Hayasaka, 2003)
% combines bootstrapping with permutation testing (Ng et al., 2014) for SVM weights 
% For each permutation, we draw B bootstrap samples with replacement and apply SVM. B was set to 1000 to keep computational time practical. Denoting 
% the classifier weights for each bootstrap sample b as Wb, we compute the normalized mean over bootstrap samples:
% W_mean = 1/B * Σb(Wbpq/std(Wb).
% 相当于每次bootstrap后算出来的w，除以标准差进行标准化，然后算平均的w. 然后保留均值w中最大的那一个
% Milazzo, A.C., et al 2016.

% if ~exist('perm_times','var') || isempty(perm_times)
%     % perm_times 参数为空，或者不以变量的形式存在；
%     perm_times = 1e4;
% end
disp('-------------------Running bootstrapping with permutation testing-------------');
tic;

if nargin < 3, perm_times = 1e4;end
disp(perm_times);

sample_num = size(label,1);
train_data = insts;
% [real_data,predicted_data]  = deal(zeros(1,perm_times));
[W_max,W_min] = deal(zeros(perm_times,1));
Dist_h = zeros(perm_times, 1);
% permutation test
parfor i = 1:perm_times
    seed = randperm(sample_num);
    train_data = insts;
    train_data(seed(1),:) = []; seed_data = seed(2:end); % 去除一个样本，保持和交叉验证一致的样本数
%     test_data = insts(seed(1),:);这些是看置换后训练出的模型的表现
%     test_label = label(seed(1));
    label_permed = label(seed_data);
    
    % bootstrap for 1e3 times
    B = 1e3;
    weights_boot = zeros(size(insts,2), B); % features * B大小的权重矩阵
    for j = 1:B
        num_seed = numel(label) - 1;
        bootseed = randsample(num_seed,num_seed,true);
        train_booted = train_data(bootseed,:);% 从1-47中取47个数字，有放回去
        label_pt_booted = label_permed(bootseed,1); % 取对应的置换后的label（不能不对应，不然同样的样本对应不同的lable）
        model = svmtrain(label_pt_booted,train_booted, '-s 3 -t 0 -q'); % -q,quiet
        weights = Cal_svm_paras(model);
        weights_boot(:,j) = weights;
    end
    % W_mean = 1/B * Σb(Wbpq/std(Wb).
    W_mean = mean(weights_boot./std(weights_boot),2); % 获取B次bootstrap后的平均值
    % W_mean = (weights_boot * (1./std(weights_boot))')/1000;
    % W矩阵
%     % store the correlation value to evaluate p value 
%     Dist_h(i) = corr(label_permed, train_data * W_mean);
    
    % store the maximum element of W_mean for each permutation.
    W_max(i) = max(W_mean); % 储存最大的那个权重
    W_min(i) = min(W_mean); % 储存最大的那个权重（负的）% 取绝对值后不需要，只保留强度信息。
    
%     [~,~,dec_value] = svmpredict(label(seed(1)), insts(seed(1),:),model);
%     real_data(i) = label(seed(1));
%     predicted_data(i) = dec_value;
%     获取features * 1的权重值 
end

% h = corr(real_data',predicted_data'); % 经过一万次模拟，出来的r值-0.19，r^2不到0.04，很低。
Dist_max = W_max;
Dist_min = W_min;
toc;
disp('Done')
end

