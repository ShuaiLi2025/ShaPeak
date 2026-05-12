% demon recovery problems  
clc; close all; clear all; warning off;
addpath(genpath(pwd));

n         = 1000; 
m         = ceil(0.5*n);
s         = ceil(0.1*n);  
nf        = 0.00;
p         = 2;

T         = randperm(n,s);  
xopt      = zeros(n,1);  
xopt(T)   = 1;  
data.A    = randn(m,n)/sqrt(m);
data.b    = data.A(:,T)*xopt(T)+nf*randn(m,1); 
acc       = @(x)norm(xopt-x)/norm(xopt);
fucf      = @(x)norm(data.A*x-data.b)^2/2;
res       = []; 

sn        = s/n;
it0       = max(10,2*ceil(sn*100/(p-1)));
r0        = 5/10^(sn*10)*100^(2-p);
sigma0    = (0.06-sn/10)*10^(p-2);
if m/n   >= 0.6 
   sigma0 = min(5e-1,sigma0*100);  
   it0    = max(10,ceil(it0/2));
end
pars.preQ  = 1; 
pars.sigma = sigma0;  
pars.it0   = it0;
pen        = r0*norm(data.b'*data.A,'f')/sqrt(n);
func       = @(x,key)funRecovery(x,key,p,data);
SPFs       = {'gaa2205','haa2205'};
for i         = 1:length(SPFs)
    pars.penf = SPFs{i}; 
    out       = ShaPeakADMM(func,n,pen,pars); 
    res       = [res; acc(out.sol) fucf(out.sol) out.iter out.time]
end

