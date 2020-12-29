%%%%%% Predict behaviour impairment with  resting-state functional connectivity  %%%%%%%%%%%%%%%%%%%%%%
%%%%%% Functional connectivity analysis performed by conn (functional connectivity toolbox) 19c. %%%%%%
%%%%%% Prediction tool: libsvm 3.23.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;close;
%% Preparation: load data 
load beh_data
Ftest_file = 'F_results_withoutGSR.mat';
% Ftest_file = 'F_results_withGSR.mat';
load(Ftest_file);
[ROI_FCmat, ROI_Pmat] = deal(zeros([numel(ROI), size(ROI(1,1).y)]), zeros(numel(ROI), numel(ROI)));
ROI_FCmat(:,:,numel(ROI) + 1:end,:) = [];
for i = 1:numel(ROI)
    % load FC values and P values of F test.
    % contrast matrix: [-1 1 0; 0 -1 1];
    [FCvalues, Pvalues] = deal(ROI(1,i).y, ROI(1,i).p);
    [FCvalues, Pvalues] = deal(FCvalues(:,1:numel(ROI), :), Pvalues(1:numel(ROI)));
    [ROI_FCmat(i, :, :, :), ROI_Pmat(i, :)] = deal(FCvalues, Pvalues);
end

ROI_FCmat = permute(ROI_FCmat, [2,4,1,3]); % permute matrix dimension to Sub * Session * ROI * ROI;

%% Extract features (ROI-level FDR corrected significant results) and behaviour data
Q = conn_fdr(ROI_Pmat, 2 ,0.05); % FDR correction
Sig_links = logical(Q + Q');

% calculate FC values of session 2 minus session 1  & session 3 minus session 2 
[FC_diffs21, FC_diffs32] =  deal(squeeze(ROI_FCmat(:, 2, :, :) - ROI_FCmat(:,1, :, :)), ...
    squeeze(ROI_FCmat(:, 3, :, :) - ROI_FCmat(:,2, :, :))); 

Sig_links = triu(Sig_links, 1); % preserve the upper triangle part
[ROI_index(:,1), ROI_index(:,2)] = find(Sig_links == 1); % acquire ROI index in power 264 template

links_num = numel(ROI_index(:,1));
disp([num2str(links_num), ' significant functional connectivities as feature values']);

% load power 264 template info and info of significant links
load powerInfo

[sigROI1net, sigROI2net] = deal(powerNet(ROI_index(:,1)), powerNet(ROI_index(:,2)));
[sigROI1MNI, sigROI2MNI] = deal(powerMNI(ROI_index(:,1), :), powerMNI(ROI_index(:,2), :));
% subjects * diffs of significant links
[FC_diffs21, FC_diffs32] = deal(FC_diffs21(:, Sig_links), FC_diffs32(:, Sig_links)); 

% throw invalid subjects to match behaviour data 
FC_diffs21(invalid_subs, :) = []; FC_diffs32(invalid_subs, :) = [];



% extract behaviour differences of 23 subjects
[RT_diffs21, RT_diffs32] = deal(PVT_RT(:, 7) - PVT_RT(:, 1), PVT_RT(:, 9) - PVT_RT(:, 7));

[sleepiness_diffs21, sleepiness_diffs32] = deal(sleepiness(:, 2) - sleepiness(:, 1), ...
    sleepiness(:, 3) - sleepiness(:, 2)) ;

% Pull these values into vectors
[Features, depend_RT, depend_sleepiness] = deal([FC_diffs21; FC_diffs32], ...
    [RT_diffs21; RT_diffs32], [sleepiness_diffs21; sleepiness_diffs32]);

%% memory release
clear comfort FC_diffs21 FC_diffs32 FCvalues Ftest_file i invalid_subs ...
     planning Pvalues PVT_RT Q ROI ROI_FCmat ROI_Pmat...
     RT_diffs21 RT_diffs32 Sig_links sleepiness sleepiness_diffs21 sleepiness_diffs32
%% prediction with support vector machine (by linear kernel), leave-one-out cross validation
model = svmtrain(depend_RT,Features, ['-s 3 -t 0 -q -v ', num2str(numel(depend_RT))]);

model = svmtrain(depend_sleepiness,Features, ['-s 3 -t 0 -q -v ', num2str(numel(depend_RT))]);

%% Permutation & bootstrap to identify significant functional connectivities
dependentNames = {'PVT_RT';'ARSQ_sleepiness'};
dependentVars = {depend_RT; depend_sleepiness};
VarsNum = numel(dependentVars);
% initial a struct to store model stats detailed results
Model_stats = repmat(struct('name', NaN, 'RawBeh', NaN, 'Dist_max', NaN, 'Dist_min',NaN, ...
    'weights', NaN,'decValue', NaN,'meanWeights', NaN,'Pos_links',NaN, 'Neg_links', NaN, ...
    'ROI1Network', NaN, 'ROI2Network', NaN, 'ROI1MNIspace', NaN, 'ROI2MNIspace', NaN), VarsNum, 1);
tic;
for n = 1:VarsNum % loop for each dependentVars
    Model_stats(n,1).name = dependentNames{n};
    Model_stats(n,1).RawBeh = dependentVars{n};
    [Model_stats(n,1).Dist_max,Model_stats(n,1).Dist_min, Dist_h] = bootstrap_PT(Model_stats(n,1).RawBeh,Features, 1e2);

    % cal svm model weights and decision value of each Cross validation
    [Model_stats(n,1).weights, Model_stats(n,1).decValue] = Extract_Model_Details_CV(Model_stats(n,1).RawBeh, Features);
    [h,p] = corr(Model_stats(n,1).decValue,Model_stats(n,1).RawBeh);
    % find significant links
    Model_stats(n,1).meanWeights = mean(Model_stats(n,1).weights, 2); % calculate mean weights of all Cross folds validation
    Model_stats(n,1).Pos_links = find(Model_stats(n,1).meanWeights >= prctile(Model_stats(n,1).Dist_max,97.5)); % 99.95%
    Model_stats(n,1).Neg_links = find(Model_stats(n,1).meanWeights <= prctile(Model_stats(n,1).Dist_min,2.5)); % 0.05%
    siglinksPT = [Model_stats(n,1).Pos_links; Model_stats(n,1).Neg_links];
    % load networks and MNI coordinates of PT significant links 
    Model_stats(n,1).ROI1Network = sigROI1net(siglinksPT);
    Model_stats(n,1).ROI2Network = sigROI2net(siglinksPT);
    Model_stats(n,1).ROI1MNIspace = sigROI1MNI(siglinksPT, :);
    Model_stats(n,1).ROI2MNIspace = sigROI2MNI(siglinksPT, :);

end
toc;
%% done

%% get p value of prediction

distc = Permtest(depend_RT, Features);
pvalue(1) = sum(distc > Model_stats(1,1).Correlation)/1e4;

distc = Permtest(depend_sleepiness, Features);
pvalue(2) = sum(distc > Model_stats(2,1).Correlation)/1e4;

