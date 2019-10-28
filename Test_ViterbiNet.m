% ViterbiNet example code - ISI channel with AWGN
clear all;
close all;
clc;

rng(1);

%% Parameters setting
s_nConst = 2;       % Constellation size (2 = BPSK)
s_nMemSize = 4;     % Number of taps
s_fTrainSize = 5000; % Training size
s_fTestSize = 50000; % Test data size

s_nStates = s_nConst^s_nMemSize;

v_fSigWdB=   -6:2:10;  %Noise variance in dB

s_fEstErrVar = 0.1;   % Estimation error variance
% Frame size for generating noisy training
s_fFrameSize = 500;
s_fNumFrames = s_fTrainSize/s_fFrameSize;

v_nCurves   = [...          % Curves
    1 ...                   % Deep Viterbi - perfect CSI
    1 ....                  % Deep Viterbi - CSI uncertainty
    1 ...                   % Viterbi - perfect CSI
    1 ...                   % Viterbi - CSI uncertainty
    ];


s_nCurves = length(v_nCurves);

v_stProts = strvcat(  ...
    'ViterbiNet, perfect CSI', ...
    'ViterbiNet, CSI uncertainty',...
    'Viterbi, perfect CSI', ...
    'Viterbi, CSI uncertainty');

s_nMixtureSize = s_nStates;

%% Simulation loop
v_fExps =  0.1:0.1:2;
m_fSERAvg = zeros(length(v_nCurves),length(v_fSigWdB));

for eIdx=1:length(v_fExps)
    % Exponentailly decaying channel
    v_fChannel = exp(-v_fExps(eIdx)*(0:(s_nMemSize-1)));
    
    m_fSER = zeros(length(v_nCurves),length(v_fSigWdB));
    
    % Noisy channels for CSI uncertainty
    m_fNoisyChanneltest = repmat(fliplr(v_fChannel),s_fTestSize,1) + ...
        sqrt(s_fEstErrVar)*randn(s_fTestSize,s_nMemSize);  
    
    
    % Generate training labels
    v_fXtrain = randi(s_nConst,1,s_fTrainSize);
    v_fStrain = 2*(v_fXtrain - 0.5*(s_nConst+1));
    m_fStrain = m_fMyReshape(v_fStrain, s_nMemSize);
    
    % Training with perfect CSI
    v_Rtrain = fliplr(v_fChannel) * m_fStrain;
    % Training with noisy CSI
    v_Rtrain2 = zeros(size(v_Rtrain));
    for kk=1:s_fNumFrames
        Idxs=((kk-1)*s_fFrameSize + 1):kk*s_fFrameSize;
        v_Rtrain2(Idxs) =  fliplr(v_fChannel + sqrt(s_fEstErrVar)*randn(size(v_fChannel))*diag(v_fChannel)) ...
            * m_fStrain(:,Idxs);
    end
    
    
    % Generate test labels
    v_fXtest = randi(s_nConst,1,s_fTestSize);
    v_fStest = 2*(v_fXtest - 0.5*(s_nConst+1));
    m_fStest= m_fMyReshape(v_fStest, s_nMemSize);
    v_Rtest = fliplr(v_fChannel) * m_fStest;
    
    % Loop over number of SNR
    for mm=1:length(v_fSigWdB)
        s_fSigmaW = 10^(-0.1*v_fSigWdB(mm));
        % LTI AWGN channel
        v_fYtrain = v_Rtrain + sqrt(s_fSigmaW)*randn(size(v_Rtrain));
        v_fYtrain2 = v_Rtrain2 + sqrt(s_fSigmaW)*randn(size(v_Rtrain));
        v_fYtest = v_Rtest + sqrt(s_fSigmaW)*randn(size(v_Rtest));
        
        
        % Viterbi net - perfect CSI
        if(v_nCurves(1)==1)
            % Train network
            [net, GMModel] = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize, s_nMixtureSize);
            % Apply ViterbiNet detctor
            v_fXhat =  ApplyViterbiNet(v_fYtest, net, GMModel, s_nConst, s_nMemSize);
            % Evaluate error rate
            m_fSER(1,mm) = mean(v_fXhat ~= v_fXtest);
        end
        
        % Viterbi net - CSI uncertainty
        if(v_nCurves(2)==1)
            % Train network using training with uncertainty
            [net, GMModel] = GetViterbiNet(v_fXtrain, v_fYtrain2, s_nConst, s_nMemSize, s_nMixtureSize);
            % Apply ViterbiNet detctor
            v_fXhat =  ApplyViterbiNet(v_fYtest, net, GMModel, s_nConst, s_nMemSize);
            % Evaluate error rate
            m_fSER(2,mm) = mean(v_fXhat ~= v_fXtest);
            
        end
        
        % Model-based Viterbi
        if((v_nCurves(3)+v_nCurves(4))>0)
            
            m_fLikelihood3 = zeros(s_fTestSize,s_nStates);
            m_fLikelihood4 = zeros(s_fTestSize,s_nStates);
            % Compute coditional PDF for each state
            for ii=1:s_nStates
                v_fX = zeros(s_nMemSize,1);
                Idx = ii - 1;
                for ll=1:s_nMemSize
                    v_fX(ll) = mod(Idx,s_nConst) + 1;
                    Idx = floor(Idx/s_nConst);
                end
                v_fS = 2*(v_fX - 0.5*(s_nConst+1));
                m_fLikelihood3(:,ii) = normpdf(v_fYtest' -  fliplr(v_fChannel)*v_fS,0,s_fSigmaW);
                m_fLikelihood4(:,ii) = normpdf(v_fYtest' -  m_fNoisyChanneltest*v_fS,0,s_fSigmaW);
            end
            % Apply Viterbi detection based on computed likelihoods
            v_fXhat3 = v_fViterbi(m_fLikelihood3, s_nConst, s_nMemSize);
            v_fXhat4 = v_fViterbi(m_fLikelihood4, s_nConst, s_nMemSize);
            % Evaluate error rate
            m_fSER(3,mm) = mean(v_fXhat3 ~= v_fXtest);
            m_fSER(4,mm) = mean(v_fXhat4 ~= v_fXtest);
        end
        
        % Display SNR index
        mm
    end
    m_fSERAvg = m_fSERAvg + m_fSER;
    
    % Dispaly exponent index
    eIdx
end
m_fSERAvg = m_fSERAvg/length(v_fExps);


%% Display results
v_stPlotType = strvcat( '-rs', '--go', '-.b^',  ':kx',...
    '-g<', '-g*', '-m>', '-mx', '-c^', '-cv');

v_stLegend = [];
fig1 = figure;
set(fig1, 'WindowStyle', 'docked');
%
for aa=1:s_nCurves
    if (v_nCurves(aa) ~= 0)
        v_stLegend = strvcat(v_stLegend,  v_stProts(aa,:));
        semilogy(v_fSigWdB, m_fSERAvg(aa,:), v_stPlotType(aa,:),'LineWidth',1,'MarkerSize',10);
        hold on;
    end
end

xlabel('SNR [dB]');
ylabel('Symbol error rate');
grid on;
legend(v_stLegend,'Location','SouthWest');
hold off;



