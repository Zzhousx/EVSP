clear;
clc;
load('fs_jaffe.mat'); 
warning('off');
rng('default');

X=fea;
Y=gnd;
[n,d]=size(X);
X_nor=NormalizeFea(X);
sizep=4;
maxFES = 40;
alpha=100;
beta=100;
[time,ACC,NMI] = EVSP(X_nor,gnd,maxFES,sizep,alpha,beta);
