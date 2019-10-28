function m_fMat= m_fMyReshape(v_fVec, s_nRows)

% Reshape vector into matrix form with interleaved columns
%
% Syntax
% -------------------------------------------------------
% m_fMat = m_fMyReshape(v_fVec, s_nRows)
%
% INPUT:
% -------------------------------------------------------
% v_fVec - vector to reshape
% s_nRows - number of rows in matrix represetnation
% 
%
% OUTPUT:
% -------------------------------------------------------
% m_fMat - matrix represetnation



s_nCols = length(v_fVec);
v_fVec = reshape(v_fVec,1,s_nCols);

m_fMat = ones(s_nRows, s_nCols);

for kk=1:s_nRows
    ll=s_nRows-kk+1;
    m_fMat(ll,1:end-ll+1) = v_fVec(ll:end);
end

