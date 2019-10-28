function [net, GMModel] = GetViterbiNet(v_fXtrain, v_fYtrain ,s_nConst, s_nMemSize, s_nMixtureSize)

% Generate and train a new ViterbiNet conditional distribution network
%
% Syntax
% -------------------------------------------------------
% [net, GMModel] = GetViterbiNet(m_fXtrain,v_fYtrain ,s_nConst)
%
% INPUT:
% -------------------------------------------------------
% v_fXtrain - training symobls vector
% v_fYtrain - training channel outputs (vector with training size entries)
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% s_nMixtureSize - finite mixture size for PDF estimator (positive integer)
%
%
% OUTPUT:
% -------------------------------------------------------
% net - trained neural network model
% GMModel - trained mixture model PDF estimate

% Reshape input symbols into a matrix representation
m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize);
 

% Generate neural network
inputSize = 1;
numHiddenUnits = 100;
numClasses = s_nConst^s_nMemSize;


% Work around converting an LSTM, which is the supported first layer for seuquence proccessing networks in Matlab, into a perceptron with sigmoid activation
LSTMLayer = lstmLayer(numHiddenUnits,'OutputMode','last'... 
    , 'RecurrentWeightsLearnRateFactor', 0 ...
    , 'RecurrentWeightsL2Factor', 0 ...
    );
LSTMLayer.RecurrentWeights = zeros(4*numHiddenUnits,numHiddenUnits);

% Generate network model
layers = [ ...
    sequenceInputLayer(inputSize)
    LSTMLayer
    fullyConnectedLayer(floor(numHiddenUnits/2))
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


% Train network with default learning rate
net = TrainViterbiNet(m_fXtrain,v_fYtrain ,s_nConst, layers, 0);

% Compute output PDF using GMM fitting
GMModel = fitgmdist(v_fYtrain',s_nMixtureSize,'RegularizationValue',0.1);
