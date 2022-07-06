clc;clear

basedir = 'Directory with raw ECG as .mat';
targetdir = 'Directory on which filtered ECG will be saved';

FileDirs = struct2cell(dir(strcat(basedir, '*.mat')));
FileNames = cellfun(@(x) x(1:end-4), FileDirs(1,:), 'UniformOutput', false);

GeneratedDirs = struct2cell(dir(strcat(targetdir, '*.csv')));
GeneratedNames = cellfun(@(x) x(1:end-4), GeneratedDirs(1,:), 'UniformOutput', false);

TargetNames = setdiff(FileNames, GeneratedNames)
display(size(TargetNames, 2))


for i = 1:size(TargetNames, 2)

    filename = TargetNames{i};
    display(i)
    dir = strcat(basedir, TargetNames{i}, '.mat');
    signalinfo = load(dir);
    signal = signalinfo.signal;
    Fs = 500;
    DenoisingData = zeros(size(signal));
    
    for j=1:12
        OrigECG  = signal(:,j);
        %fp=50;fs=60;
        %rp=1;rs=2.5;
        %wp=fp/(Fs/2);ws=fs/(Fs/2);
        %[n,wn]=buttord(wp,ws,rp,rs);
        %[bz,az] = butter(n,wn);
        LPassDataFile=OrigECG;
        
        t = 1:length(LPassDataFile);
        yy2 = smooth(t,LPassDataFile,0.1,'rloess');
        BWRemoveDataFile = (LPassDataFile-yy2);
        Dl1=BWRemoveDataFile;
        
        for k=2:length(Dl1)-1
            Dl1(k)=(2*Dl1(k)-Dl1(k-1)-Dl1(k+1))/sqrt(6);
        end
        NoisSTD = 1.4826*median(abs(Dl1-median(Dl1)));
        DenoisingData(:,j)= NLM_1dDarbon(BWRemoveDataFile,(1.5)*(NoisSTD),5000,10);
    end
    writematrix(DenoisingData, strcat(targetdir, filename, '.csv'))

end
    

function [denoisedSig,debug] = NLM_1dDarbon(signal,lambda,P,PatchHW)

if length(P)==1  % scalar has been entered; expand into patch sample index vector
    Pvec = -P:P;
else
   Pvec = P;  % use the vector that has been input  
end
debug=[];
N = length(signal);

denoisedSig = NaN*ones(size(signal));

% to simpify, don't bother denoising edges
iStart=1+PatchHW+1;
iEnd = N-PatchHW;
denoisedSig(iStart:iEnd) = 0;

debug.iStart = iStart;
debug.iEnd = iEnd;

% initialize weight normalization
Z = zeros(size(signal));
cnt = zeros(size(signal));    

% convert lambda value to 'h', denominator, as in original Buades papers
Npatch = 2*PatchHW+1;
h = 2*Npatch*lambda^2;

for idx = Pvec  % loop over all possible differences: s-t
    % do summation over p  - Eq. 3 in Darbon
    k=1:N;
    kplus = k+idx;
    igood = find(kplus>0 & kplus<=N);  % ignore OOB data; we could also handle it
    SSD=zeros(size(k));
    SSD(igood) = (signal(k(igood))-signal(kplus(igood))).^2;
    Sdx = cumsum(SSD);
   
    for ii=iStart:iEnd  % loop over all points 's'
        distance = Sdx(ii+PatchHW) - Sdx(ii-PatchHW-1); % Eq 4; this is in place of point-by-point MSE
        % but note the -1; we want to icnlude the point ii-iPatchHW

        w = exp(-distance/h);  %Eq 2 in Darbon
        t = ii+idx;  % in the papers, this is not made explicit
        
        if t>1 && t<=N
            denoisedSig(ii) = denoisedSig(ii) + w*signal(t);
            Z(ii) = Z(ii) + w;
            cnt(ii) = cnt(ii)+1;
        end

    end
end % loop over shifts

% now apply normalization
denoisedSig = denoisedSig./(Z+eps);
denoisedSig(1:PatchHW+1) =signal(1:PatchHW+1);
denoisedSig(end-PatchHW+1:end) =signal(end-PatchHW+1:end);
debug.Z = Z;

end


