function Out = ShaPeakADMM(Funcs,n,pen,pars)
% This code aims at solving the 0/1 regularized optimization with form
%
%         min_{x\in 0<=x<=1} F(x) := f(x) + pen*sum_i g(x_i),
%
% where g is a sharp-peak function
%       pen > 0 it the penalty parameter 
%       
%--------------------------------------------------------------------------
% Inputs:
%     func:  A function handle defines (objective,gradient) of f  (required)
%     n   :  Dimension of the solution x                          (required)
%     pen :  Penalty parameter                                    (required) 
%     pars:  Parameters are all OPTIONAL
%            pars.x0      --  Starting point of x,   pars.x0 = zeros(n,1)  (default)
%            pars.prob    --  \in {'recovery','mimo','1bitmimo','qubo', or 'None'}  
%            pars.preQ    --  =0, using first order approximation  
%                             =1, using second order approximation         (default)
%            pars.show    --  Display results or not for each iteration    (default, 1)
%            pars.maxit   --  Maximum number of iterations                 (default,2000) 
%            pars.tol     --  Tolerance of the halting condition           (default,1e-6)
%
% Outputs:
%     Out.sol :   The sparse solution x
%     Out.time:   CPU time
%     Out.iter:   Number of iterations
%     Out.obj :   Objective function value at Out.sol 
%--------------------------------------------------------------------------
% This code is programmed based on the algorithm proposed in 
% "Shenglong Zhou, Shuai Li, Hui Zhang, and Ziyan Luo, 
%  Sharp-Peak Functions for Exactly Penalizing Binary Integer Programming, 2024."
% Send your comments and suggestions to <<< slzhou2021@163.com >>> 
% Warning: Accuracy may not be guaranteed !!!!! 
%--------------------------------------------------------------------------

warning off;
t0  = tic;
if  nargin < 3
    disp(' No enough inputs. No problems will be solverd!'); return;
end
if nargin < 4; pars = [];  end 

[x,tol,maxit,show,preQ,sigma,it0,check,penrt,penfun,proxfun,grate,gfreq,drate,dfreq] = setparameters(n, pars);

Fnorm  = @(var)norm(var,'fro')^2; 
 
w      = x;
y      = zeros(n,1); 
Error  = zeros(maxit,1);
obj    = Funcs(x,'f');
Objfx  = obj*ones(maxit,1);
pen0   = pen;
mark   = 0;
n0     = 6e3;
InQ    = sparse(n,n);
if  preQ==1 || preQ==2 
    Hw   = Funcs(w,'H'); 
    flag = isa(Hw,'function_handle');
    if  flag || n>n0 
        if  flag
            InQ = @(sig,v)(sig*v+Hw(v));  
        else
            InQ = @(sig,v)(sig*v+Hw*v); 
        end
    else
        InQ = inv(sigma*speye(n)+Hw); 
    end 
end

IsfunInQ = ~isa(InQ,'function_handle') && preQ~=0;


if  show
    fprintf(' \n Start to run the solver -- ShaPeak \n');
    fprintf(' ----------------------------------------------\n');
    fprintf(' Iter       Error        Binary       Time(sec) \n'); 
    fprintf(' ----------------------------------------------\n');
end

% The main body
for iter = 1:maxit    

    gyy    = Funcs(w,'g') + y;
    if  preQ == 0
        x = w - gyy/sigma;  
    elseif IsfunInQ
        x = w - InQ*gyy;
    else
        x = w - my_cg(@(v)InQ(sigma,v),gyy,1e-8,6,zeros(n,1)) ;
    end
    w0   = w;
    w    = proxfun(x+y/sigma,pen/sigma);
    xy   = x-w;  
    y    = y+sigma*xy; 
    sx   = penfun(w);

    Error(iter) = max([Fnorm(xy),Fnorm(w-w0)])/(1+Fnorm(w));
    if  show  && (mod(iter,10)==0 || iter<10)
        fprintf('%4d       %5.2e     %7.2e       %6.3f\n',...
        iter, Error(iter), sx, toc(t0)); 
    end
              
    % Stopping criteria 
    if sx <=0 && Error(iter) <tol
        if  show  && mod(iter,10)~=0
            fprintf('%4d       %5.2e     %7.2e       %6.3f\n',...
            iter, Error(iter), sx, toc(t0)); 
        end
        check = check || (iter>2e3);
        if  check && n<=1e4
            z   = CheckOpt(w,n,Funcs,Funcs(w,'f'));
            pen = pen0/log(iter);
            if nnz(z-w)==0; break; end
            w   = z;
        else
            break;
        end
    end
    
    mark = mark + (sx==0);
    if  mod(iter,it0)==0 && sx>0 
        pen = pen*penrt;   
    end
 
    if  mark>5 && Error(iter) > tol && mod(iter,gfreq)==0   
        sigma  = min(1e4,sigma*grate); 
        change = 1;
    elseif sx >0 && Error(iter) <tol && mod(iter,dfreq)==0
        sigma  = max(1e-3,sigma/drate);
        pen    = pen*penrt;       
        change = 1;
    else
        change = 0;
    end
 
    if change && IsfunInQ 
       InQ = inv(sigma*speye(n)+Hw);  
    end
    
end

fprintf(' ----------------------------------------------\n');
Out.time    = toc(t0);
Out.iter    = iter;
Out.sol     = w;
Out.obj     = Funcs(w,'f');  
Out.Error   = Error; 
end

% Set up parameters -------------------------------------------------------
function [x0,tol,maxit,show,preQ,sigma,it0,check,penrt,penfun,proxfun,grate,gfreq,drate,dfreq] = setparameters(n, pars)
	
    if  isfield(pars,'penrt');penrt = pars.penrt; else; penrt = 1.25;     end    
    if isfield(pars,'x0');    x0    = pars.x0;    else; x0 = 0*ones(n,1); end 
    if isfield(pars,'tol');   tol   = pars.tol;   else; tol= 1e-5*sqrt(n);end  
    if isfield(pars,'it0');   it0   = pars.it0;   else; it0   = 100;      end    
    if isfield(pars,'show');  show  = pars.show;  else; show  = 1;        end 
    if isfield(pars,'check'); check = pars.check; else; check = 0;        end    
    if isfield(pars,'preQ');  preQ  = pars.preQ;  else; preQ  = 1;        end 
    if isfield(pars,'sigma'); sigma = pars.sigma; else; sigma = 0.5;      end 
    if isfield(pars,'maxit'); maxit = pars.maxit; else; maxit = 1e4;      end
    if isfield(pars,'prob');  prob  = pars.prob;  else; prob = 'none';    end
    maxit = maxit + 1e5*check;
    
    switch prob
        case 'recovery' 
            grate  = 1.2;   
            gfreq  = 10;    
            drate  = 1.1;   
            dfreq  = 100;    
        case 'mimo' 
            grate  = 1.2;   
            gfreq  = 10;    
            drate  = 1.1;  
            dfreq  = 10;
        case '1bitmimo' 
            grate  = 1.2;   
            gfreq  = 5;    
            drate  = 1.1;  
            dfreq  = 100;
        case 'qubo' 
            grate  = 1.2;   
            gfreq  = 10;    
            drate  = 1.1;  
            dfreq  = 100*(n<=2e3)+10*(n>2e3); 
        otherwise     
            grate  = 1.2;   
            gfreq  = 10;    
            drate  = 1.1;  
            dfreq  = 100;  
    end

    if isfield(pars,'penf') 
         switch pars.penf
            case 'Piecewise'
                penfun  = @(x)Piecewise(x);
                proxfun = @(z,eta)ProxPiecewise(z,eta); 
            case '111100'
                penfun  = @(x)SPF111100(x);
                proxfun = @(z,eta)ProxSPF111100(z,eta); 
            case '111101'
                penfun  = @(x)SPF111101(x);
                proxfun = @(z,eta)ProxSPF111101(z,eta); 
            case '111105'
                penfun  = @(x)SPF111105(x);
                proxfun = @(z,eta)ProxSPF111105(z,eta);
            case 'gaa2200'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFgaa2200(x,a);
                proxfun = @(z,eta)ProxSPFgaa2200(z,a,eta);     
            case 'gaa2205'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFgaa2205(x,a);
                proxfun = @(z,eta)ProxSPFgaa2205(z,a,eta); 
            case 'gaa2201'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFgaa2201(x,a);
                proxfun = @(z,eta)ProxSPFgaa2201(z,a,eta); 
            case 'haa2200'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFhaa2200(x,a);
                proxfun = @(z,eta)ProxSPFhaa2200(z,a,eta);    
            case 'haa2205'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFhaa2205(x,a);
                proxfun = @(z,eta)ProxSPFhaa2205(z,a,eta);
            case 'haa2201'
                if isfield(pars,'a'); a=pars.a; else; a=2.5; end 
                penfun  = @(x)SPFhaa2201(x,a);
                proxfun = @(z,eta)ProxSPFhaa2201(z,a,eta);    
             otherwise
                if isfield(pars,'abpq'); abpq=pars.abpq; else; abpq=[2 2 2 2]; end 
                if isfield(pars,'u');    u   =pars.u;    else; u   =0.5;       end
                if isfield(pars,'key');  key =pars.key;  else; key ='h';       end
                penfun  = @(x)SPFs(x,u,abpq,key);
                proxfun = @(z,eta)ProxSPFs(z,u,eta,abpq,key); 
         end
    else  
        penfun  = @(x)SPFhaa2205(x,2.5);
        proxfun = @(z,eta)ProxSPFhaa2205(z,2.5,eta);  
    end
    
end


% check the optimality ----------------------------------------------------
function x = CheckOpt(x,n,Funcs,fx)

     T1  = find(x==1); 
     T2  = find(x==0); 
     nx1 = length(T1);
     if nx1>n-nx1; T0=T1; T1=T2; T2=T0; end

     for t = 1:length(T1)
         y = x;
         y(T1(t)) = 1-x(T1(t));  
         if Funcs(y,'f') < fx  
             x = y; break; 
         end 
     end

     if  t== length(T1)
         for t = 1:length(T2)
             y = x;
             y(T2(t)) = 1-x(T2(t));  
             if Funcs(y,'f') < fx  
                 x = y; break; 
             end 
         end
     end
     clear T1 T2 y
end

% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    if norm(b,'fro')==0; x=zeros(size(x)); return; end
    if ~isa(fx,'function_handle'); fx = @(v)fx*v; end
    r = b;
    if nnz(x)>0; r = b - fx(x);  end
    e = norm(r,'fro')^2;
    t = e;
    p = r;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        w  = fx(p);
        pw = p.*w;
        a  = e/sum(pw(:));
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = norm(r,'fro')^2;
        p  = r + (e/e0)*p;
    end 
end