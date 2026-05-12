% demon QUBO problems  
clc; close all; clear all; warning off;
addpath(genpath(pwd));

n          = 1000;
neg_ratio  = 0.5;
dens_ratio = 0.8;

% generate data
data.Q     = dataQUBO(n, neg_ratio, dens_ratio);
data.Q     = data.Q/svds(data.Q,1);
res        = [];

pars.x0    = rand(n,1);
pars.preQ  = 0;
pars.sigma = 0.01;
pars.it0   = 10;
pars.penrt = 1.1;
pars.tol   = 1e-5*sqrt(n);
pen        = 5e-6*norm(data.Q,'f'); 
func       = @(x,key)funQUBO(x,key,data.Q);

SPFs       = {'gaa2205','haa2205'};
pars.a     = 2.5;
for i      = 1:length(SPFs)
    pars.penf = SPFs{i};   
    out       = ShaPeakADMM(func,n,1*pen,pars); 
    res       = [res; -out.obj out.iter out.time]
end
