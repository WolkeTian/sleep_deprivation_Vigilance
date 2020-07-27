function [Dist_max,Dist_min, Dist_h] = bootstrap_PT(label,insts,perm_times)
% using maximum statistics implicitly accounts for multiple comparisons (Nichols and Hayasaka, 2003)
% combines bootstrapping with permutation testing (Ng et al., 2014) for SVM weights 
% For each permutation, we draw B bootstrap samples with replacement and apply SVM. B was set to 1000 to keep computational time practical. Denoting 
% the classifier weights for each bootstrap sample b as Wb, we compute the normalized mean over bootstrap samples:
% W_mean = 1/B * ��b(Wbpq/std(Wb).
% �൱��ÿ��bootstrap���������w�����Ա�׼����б�׼����Ȼ����ƽ����w. Ȼ������ֵw��������һ��
% Milazzo, A.C., et al 2016.

% if ~exist('perm_times','var') || isempty(perm_times)
%     % perm_times ����Ϊ�գ����߲��Ա�������ʽ���ڣ�
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
    train_data(seed(1),:) = []; seed_data = seed(2:end); % ȥ��һ�����������ֺͽ�����֤һ�µ�������
%     test_data = insts(seed(1),:);��Щ�ǿ��û���ѵ������ģ�͵ı���
%     test_label = label(seed(1));
    label_permed = label(seed_data);
    
    % bootstrap for 1e3 times
    B = 1e3;
    weights_boot = zeros(size(insts,2), B); % features * B��С��Ȩ�ؾ���
    for j = 1:B
        num_seed = numel(label) - 1;
        bootseed = randsample(num_seed,num_seed,true);
        train_booted = train_data(bootseed,:);% ��1-47��ȡ47�����֣��зŻ�ȥ
        label_pt_booted = label_permed(bootseed,1); % ȡ��Ӧ���û����label�����ܲ���Ӧ����Ȼͬ����������Ӧ��ͬ��lable��
        model = svmtrain(label_pt_booted,train_booted, '-s 3 -t 0 -q'); % -q,quiet
        weights = Cal_svm_paras(model);
        weights_boot(:,j) = weights;
    end
    % W_mean = 1/B * ��b(Wbpq/std(Wb).
    W_mean = mean(weights_boot./std(weights_boot),2); % ��ȡB��bootstrap���ƽ��ֵ
    % W_mean = (weights_boot * (1./std(weights_boot))')/1000;
    % W����
%     % store the correlation value to evaluate p value 
%     Dist_h(i) = corr(label_permed, train_data * W_mean);
    
    % store the maximum element of W_mean for each permutation.
    W_max(i) = max(W_mean); % ���������Ǹ�Ȩ��
    W_min(i) = min(W_mean); % ���������Ǹ�Ȩ�أ����ģ�% ȡ����ֵ����Ҫ��ֻ����ǿ����Ϣ��
    
%     [~,~,dec_value] = svmpredict(label(seed(1)), insts(seed(1),:),model);
%     real_data(i) = label(seed(1));
%     predicted_data(i) = dec_value;
%     ��ȡfeatures * 1��Ȩ��ֵ 
end

% h = corr(real_data',predicted_data'); % ����һ���ģ�⣬������rֵ-0.19��r^2����0.04���ܵ͡�
Dist_max = W_max;
Dist_min = W_min;
toc;
disp('Done')
end

