
clc;    close all;   clear all;
fprintf('\n\n ===== The RPCA code is running  .....\n')

addpath('./PROPACK')

randn('state',0);   rand('twister',0);
load('PET.mat')
[m, n] = size(b) ;

sr = 1;                                          %%% Incomplete info.
Omega = randperm(m*n); p = round(sr*m*n);
Omega = Omega(1:p);    Omega = Omega';  b = b(Omega);
sigma = 0.001;         b = b + sigma * randn(p,1); %%%% add noise
% fid = fopen('Result.txt','w');
fprintf('\n**Vedio_Missing data:m:%3d,n:%3d,Omega:%4.2f **\n',m,n,sr);

% regularization parameter
tau = 1/sqrt(m);          %% balancing parameter
opts.minType = 'unc';     %% choose which model
mu  = 0.01;
stop.Max = 200;   stop.rule = 'TOL';

%% ================Setting parameters =====================
para.sct = 8000;   para.beta = 0.5;     para.gamma = 1.8;
para.model = 'unconstrained';    para.mu = 0.01;   para.tau = 1/sqrt(m);

EPS = [10^-2];

for is = 1:length(EPS)

    stop.eps = EPS(is);
    fprintf('\n========= %2.2e =============\n',stop.eps);
    tic
    % out = DDRS(m,n,Omega,b,para,stop);
    out = ours(m,n,Omega,b,para,stop);
    % out = Dong(m,n,Omega,b,para,stop);
    out = admm(m,n,Omega,b,para,stop);
    toc
    a1=0:5:out.iter;  a1(1) = 1; 

    
    figure(1);
    plot(a1,out.Rankr(a1),'--d',a1,out.SP(a1),'-.s',a1,out.obj(a1),'-.o','LineWidth',2.5);
    legend('Rankr','SP','obj'); 
%     title(['theta=',num2str(theta),'alphabar=',num2str(alph)])
    sparse_path=fullfile(pwd,'results\admm\sparse\');
    if ~isfolder(sparse_path)
        mkdir(sparse_path);
    end
    lower_path =fullfile(pwd,'results\admm\lowrank\');
    if ~isfolder(lower_path)
        mkdir(lower_path);
    end
    noise_path =fullfile(pwd,'results\admm\lowrank\');
    if ~isfolder(lower_path)
        mkdir(lower_noise_pathpath);
    end
    Path = 'C:\Users\54072\Desktop\何洪津\result_ours\';
    for k = 1:n
        outsp = out.Sparse(:,k);
        outLR = out.LowRank(:,k);
        outno = out.noise(:,k);
        outsparse = reshape(outsp,576,720);
        outlowrank = reshape(outLR,576,720);
        outnoise = reshape(outno,576,720);
        imwrite(mat2gray(abs(outsparse)),[sparse_path,num2str(k),'.png']);
        imwrite(mat2gray(abs(outlowrank)),[lower_path,num2str(k),'.png']);
        imwrite(mat2gray(abs(outnoise)),[lower_path,num2str(k),'.png']);
    end


    fprintf('& %d & 2.2f & %2.2f & %d & %d & %2.1f \n',out.iter, ...
        out.Rankr(end),out.SP(end),out.obj(end));
end
 
fprintf('Running is completed!\n')



