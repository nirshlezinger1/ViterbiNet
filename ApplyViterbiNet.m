function v_fXhat = ApplyViterbiNet(v_fY, net, GMModel, s_nConst, s_nMemSize)

% Apply ViterbiNet to observed channel outputs
%
% Syntax
% -------------------------------------------------------
% v_fXhat = ApplyViterbiNet(v_fY, net, GMModel, s_nConst)
%
% INPUT:
% -------------------------------------------------------
% v_fY - channel output vector
% net - trained neural network model
% GMModel - trained mixture model PDF estimate
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector

s_nStates = s_nConst^s_nMemSize;
% Use network to compute likelihood function
m_fpS_Y = predict(net,fShapeY(v_fY));
% Compute output PDF
v_fpY = pdf(GMModel, v_fYtest');
% Compute likelihoods
m_fLikelihood = (m_fpS_Y .* v_fpY)*s_nStates;        
% Apply Viterbi output layer
v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize);