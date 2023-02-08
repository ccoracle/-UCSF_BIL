% TMI paper's codes: Bayesian adaptive beamformer for robust electromagnetic 
% brain imaging of correlated sources in high spatial resolution
%
% this code is the main code to demostate the reconstruction of three
% sources in high resolution with high correlation using SBL-beam algorithm, the correlation
% among them is 0.95


clc
clear
close all

addpath('inverse_algorithms');
%% low leadfield matrix
nuts_low = load('leadfields/SEF_nuts_12mm_1728');
nuts_high = load('leadfields/SEF_nuts_2-5mm_144439.mat');
load('data.mat')

%% sensor data plot
figure('color','w')
plot(data')
title('sensor data');

%%
LF0 = nuts_high.Lp;
[nc nd nov]=size(LF0);
LF =reshape(LF0,nc,nd*nov);
LF0 = LF;
for i=1:size(LF,2)
    LF(:,i) = LF(:,i)./sqrt(sum(LF(:,i).^2));
end
LF = reshape(LF,nc,nd,nov);
LF = permute(LF,[1 3 2]);

%% sources reconstruction
ypost=data;

%% SBL to estimate the model data covariance under low resolution
[nc,nd,noc_low ] = size(nuts_low.Lp);
lf_low = nuts_low.Lp;
lf_low = reshape(lf_low,nc,nd*noc_low);
for i=1:noc_low*nd
    lf_low(:,i) = lf_low(:,i)./sqrt(sum(lf_low(:,i).^2));
end

tic
coup = 0;
ncf = 1;
vcs = 0;
[~,x,~,~]=SBL_cov(ypost,double(lf_low),150,nd,vcs,ncf,coup,ncf);
c = lf_low*spdiags(sum(x.^2,2),0,size(lf_low,2),size(lf_low,2))*lf_low';
time_low_champ = toc;
disp(['Elaspsed time for champagne with low resolution is: ', num2str(time_low_champ)])

% beamformer with high resolution leadfield
rgamma = 1e-5;
tic
[weight,~, po]= lcmv_par(double(LF),c,rgamma);
time_beam_par = toc;
disp(['Elaspsed time for beam with high resolution is: ', num2str(time_beam_par)])
disp(['low champ + high beam time is: ', num2str(time_low_champ+time_beam_par)])

weight = permute(weight,[1 3 2]);
w = reshape(weight,size(weight,1),size(weight,2)*size(weight,3));
s_beamchamp = w'*ypost;
xxx= sum(s_beamchamp.^2,2);
xx = reshape(xxx,nd,size(xxx,1)/nd);
power_beamchampc = sum(xx,1)';

%% 
figure('color','w');
subplot(121)
plot(power);title('ground truth power')
subplot(122)
plot(power_beamchampc);title('SBL-beam power');


