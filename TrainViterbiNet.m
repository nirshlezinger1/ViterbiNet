function net = TrainViterbiNet(m_fXtrain,v_fYtrain ,s_nConst, layers, learnRate)

% Train ViterbiNet conditional distribution network
%
% Syntax
% -------------------------------------------------------
% net = TrainViterbiNet(m_fXtrain,v_fYtrain ,s_nConst, layers, learnRate)
%
% INPUT:
% -------------------------------------------------------
% m_fXtrain - training symobls corresponding to each channel output (memory x training size matrix)
% v_fYtrain - training channel outputs (vector with training size entries)
% s_nConst - constellation size (positive integer)
% layers - neural network model to train / re-train
% learnRate - learning rate (poitive scalar, 0 for default of 0.01)
% 
%
% OUTPUT:
% -------------------------------------------------------
% net - trained neural network model

 
s_nM = size(m_fXtrain,1);

% Combine each set of inputs as a single unique category
v_fCombineVec = s_nConst.^(0:s_nM-1);

% format training to comply with Matlab's deep learning toolbox settings
v_fXcat = categorical((v_fCombineVec*(m_fXtrain-1))');
v_fYcat = num2cell(v_fYtrain');
 

if (learnRate == 0)
    learnRate = 0.01;
end

% Network parameters
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ... 
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false ...
    );%,'Plots','training-progress'); % This can be unmasked to display training convergence

% Train netowrk
net = trainNetwork(v_fYcat,v_fXcat,layers,options);