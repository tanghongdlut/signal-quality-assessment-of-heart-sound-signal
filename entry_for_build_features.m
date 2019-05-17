% Entry to get features for each heart sound recroding
% For automatic signal quality assessment for heart sound recording
% Written by Hong Tang, tanghong@dlut.edu.cn
% School of Biomedical Engineering, Dalian University of Technology, China
% Anyone is welcome to use the codes for acdamic purpose.
% Welcome to cite the works done by Hong Tang, https://www.researchgate.net/profile/Hong_Tang10
%  2019-05-19


clc;   clear;   close all

% read raw data
load('hs_data_original.mat')
sz=size(hs_data_original);

% Have the features previously been done partly ?
if isempty(dir('features.mat'))   % No, from the beginning
    start_i=1;
else                % Yes, to continue
    load('features.mat');
    sz=size(features);
    start_i=sz(1);
end

Binary_label=[];
Triple_label=[];

for k=start_i:sz(1)
    
    % get raw heart sound recording
    hs=hs_data_original(k,1).hs;
    fs=hs_data_original(k,1).fs;   %sampling frequency
    
    % preprocessing
    phs=pre_processing(hs,fs);
    
    % gent envelope from STFT
    enve=getEnvelopeFromSTFT(phs,fs);
    
    % get features 
    features(k,:)=get_features(phs,enve,fs);
    
    if hs_data_original(k,1).quality_label <=3
        Binary_label(k,1)=0;    % unacceptable quality
    else
        Binary_label(k,1)=1;     % acceptable quality
    end
    
    if hs_data_original(k,1).quality_label <=3
        Triple_label(k,1)=0;    % unacceptable quality
    elseif hs_data_original(k,1).quality_label ==4
        Triple_label(k,1)=1;    % good quality
    else
        Triple_label(k,1)=2;    % excellent quality
    end
    
    % save features regularly to avoid computer accidental shutdown
    if rem(k,100)==0
        save('features.mat','features','Binary_label','Triple_label');
        'saved'
    end
    
    k
    clear hs phs enve;
    
end


save('features.mat','features','Binary_label','Triple_label');

% 
% shutdown computer 
% system('shutdown -s -t 300');

