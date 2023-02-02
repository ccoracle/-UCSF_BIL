% ypost: sensor data after preprocessing
% nd: # of directions
% vcs = voxel covariance structure: 0 = scalar, 1 = diagonal, 2 = general
% coup: covaraince update methods, 0=covex boounding, 1 = EM, 2 = Mackay;
% prefer coup = 0
% ncf: noise covavariance form, 0 = scalar, 1 = heter; prefer ncf = 1
% lf_low: leadfield in low resolution
% lf_high: leadfield in high resolution

%% sbl with low resolution leadfield
nem = 100; %# of iterations for sbl
rgamma = 1e-5;
coup = 0;
ncf = 1;

[gamma,x,w,c]=SBL_cov(ypost,double(lf_low),nem,nd,vcs,1,coup,ncf);

%% two ways to set the model data covariance first way:
v = real(v);
Ess = diag(v);
Ryy_champ=lf_low*Ess*lf_low'+1e-5*max(eig(lf_low*Ess*lf_low'))*eye(size(lf_low*Ess*lf_low'));

%% second way
Ryy_champ = c;

%% beamformer with leadfield in high resolution
[weight,~, po]= lcmv_par(double(LF_high),Ryy_champ,0);
weight = permute(weight,[1 3 2]);
w = reshape(weight,size(weight,1),size(weight,2)*size(weight,3));
x = w'*ypost; 
