function w = extract_svm_paras(model)
% extract each features' weight of svm model
weights = model.sv_coef' * model.SVs;
w  = weights';
end
