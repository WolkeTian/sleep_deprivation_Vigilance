function Dist_corr = Permtest(label,insts,perm_times)
% if ~exist('perm_times','var') || isempty(perm_times)
%     % perm_times 参数为空，或者不以变量的形式存在；
%     perm_times = 1e4;
% end
disp('-------------------Running permutation testing-------------');
% tic;

if nargin < 3, perm_times = 1e4;end
disp(perm_times);

Dist_corr = zeros(perm_times,1);
sample_num = size(label,1);
train_data = insts;
% permutation test
for i = 1:perm_times
    seed = randperm(sample_num); 
    train_label = label(seed,:); % permute label

    expression = "svmtrain(train_label,train_data, ['-s 3 -t 0 -q -v ', num2str(numel(train_label))])";
    % load cv results
    results = evalc(expression);
    startindex = regexp(results, 't = 0') + 4;
    endindex = regexp(results, '\n\na') - 1;

    Dist_corr(i) = str2double(results(startindex:endindex));
end
disp('Done')
end

