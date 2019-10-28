function v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize)

% Apply Viterbi detection from computed priors
%
% Syntax
% -------------------------------------------------------
% v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize)
%
% INPUT:
% -------------------------------------------------------
% m_fPriors - evaluated likelihoods for each state at each time instance
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector



s_nDataSize = size(m_fPriors, 1);
s_nStates = s_nConst^s_nMemSize;
v_fXhat = zeros(1, s_nDataSize);

% Generate trellis matrix
m_fTrellis = zeros(s_nStates,s_nConst);
for ii=1:s_nStates
    Idx = mod(ii -1, s_nConst^(s_nMemSize-1));
    for ll=1:s_nConst
        m_fTrellis(ii,ll) = s_nConst*Idx + ll;
    end
    
end

% Apply Viterbi 
m_fCost = -log(m_fPriors);
v_fCtilde = zeros(s_nStates,1);

for kk=1:s_nDataSize
    m_fCtildeNext = zeros(s_nStates,1);
    for ii=1:s_nStates
        v_fTemp = zeros(s_nConst,1);
        for ll=1:s_nConst
            v_fTemp(ll) = v_fCtilde(m_fTrellis(ii,ll)) + m_fCost(kk,ii);
        end
        m_fCtildeNext(ii) = min(v_fTemp);
    end
    v_fCtilde = m_fCtildeNext;
    [~, I] = min(v_fCtilde);
    % return index of first symbol in current state
    v_fXhat(kk) = mod(I-1,s_nConst)+1;
end