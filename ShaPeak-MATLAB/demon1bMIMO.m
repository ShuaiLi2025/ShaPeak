% demon one-bit MIMO problems
clc; close all; clear all; addpath(genpath(pwd));

M         = 1000;
N         = 500;
scale     = sqrt(2);
Ntrials   = 3;

% SNR range
SNRdB      = 0:5:20; % SNR range
SNR        = 10.^(SNRdB/10);
sigma_C    = 2*N./SNR;

sigma_real = sigma_C/2;
sigma_OE   = sqrt(sigma_real)+0.5; % over-estimated noise standard variance

% QPSK constellation
mod_order = 2;
theta     = pi/(2^mod_order);

BER_ZF     = zeros(length(SNR),Ntrials);
BER_SPg    = zeros(length(SNR),Ntrials);
BER_SPh    = zeros(length(SNR),Ntrials);

Time_SPg    = zeros(length(SNR),Ntrials);
Time_SPh    = zeros(length(SNR),Ntrials);

wb = waitbar(0,'plez wait');
for i = 1:Ntrials
    fprintf('\n')
    display(['ntrials:' int2str(i)]);
    
    waitbar(i/Ntrials,wb);
    % generate channels
    H=(randn(M,N)+1i*randn(M,N))/scale;
    H_pinv=pinv(H);
    
    % generating symbols
    Databits     = round(rand(mod_order,N));
    symbol_index = bin2dec(char(Databits+48)');
    symbol_mat   = sqrt(2)* pskmod(symbol_index,2^mod_order,theta);
    v_symbol     = symbol_decode(symbol_mat,mod_order,theta,'BER');
    
    % generating noise
    n_ch = (randn(M,1)+1i*randn(M,1))/scale;
    
    for j = 1:length(SNR)
            n        = sqrt(sigma_C(j))*n_ch;
            y        = H*symbol_mat+n;
            y_r      = [real(y);imag(y)]; 
            H_tilde  = [real(H), -imag(H);imag(H), real(H)];
            y_tilde  = sign(y_r);
                     
            %  ----- ZF -----------------
            y_bit_com     = sign(real(y))+1i*sign(imag(y));
            x_zf          = H_pinv*y_bit_com;
            Bit_zf        = symbol_decode(x_zf,mod_order,theta,'BER');
            BER_ZF(j,i)   = length(find(Bit_zf-v_symbol));
            
           % ------  ShaPeakg --------------
            G_matrix   = (diag(y_tilde)*H_tilde).';
            Homega     = G_matrix/sigma_OE(j);
            data.A     = 2*H_tilde;
            At         = data.A';
            data.b     = H_tilde*ones(2*N,1)+y_r;
            pars.prob  = '1bitmimo'; 
            pars.preQ  = 1;
            

            pars.sigma = N/1e3;
            pars.it0   = 10;
            pars.penrt = 1.25+0.25*(N>=5e3);
            pars.tol   = 1e-5*sqrt(2*N);
            r          = 1e-3*(1+4*(N>=5e3))/log10(N);
            pen        = r*norm(y_tilde'*H_tilde ,'inf');
            
            HH         = Homega*Homega';
            Ht         = Homega';
            func       = @(x,key)fun1bMIMO(x,key,Ht,HH);
            pars.a     = 2.5;
            pars.penf  = 'gaa2205';
            out        = ShaPeakADMM(func,2*N,pen,pars);
            x_SPg         = 2*out.sol - 1;
            x_SPg         = x_SPg(1:N) + 1i*x_SPg(N+1:2*N);
            Bit_SPg       = symbol_decode(x_SPg,mod_order,theta,'BER');
            BER_SPg(j,i)  = length(find(Bit_SPg-v_symbol));  
            Time_SPg(j,i) = out.time;
           % ------  ShaPeakh --------------
            pars.penf  = 'haa2205';
            out        = ShaPeakADMM(func,2*N,pen,pars);
            x_SPh         = 2*out.sol - 1;
            x_SPh         = x_SPh(1:N) + 1i*x_SPh(N+1:2*N);
            Bit_SPh       = symbol_decode(x_SPh,mod_order,theta,'BER');
            BER_SPh(j,i)  = length(find(Bit_SPh-v_symbol)); 
            Time_SPh(j,i) = out.time;            
            
    end
    
end
close(wb)


% show BER results
%---- ZF ---------
BER_ZF_avg   = mean(BER_ZF,2)/(N*mod_order);
%---- shapeakg -------
BER_SPg_avg  = mean(BER_SPg,2)/(N*mod_order);
Time_SPg_avg = median(Time_SPg,2);
%---- shapeakh -------
BER_SPh_avg  = mean(BER_SPh,2)/(N*mod_order);
Time_SPh_avg = median(Time_SPh,2);


%--------- plot BER curve ----------------
figure('Renderer', 'painters','Position', [300 200 540 460])
axes('Position', [0.12 0.12 0.84 0.825])

res = [BER_ZF_avg BER_SPg_avg BER_SPh_avg]; 

colors = {'#e9162d', '#f700f3', '#1fb819'};
markers = {'o', 's', '*'};
lineStyles = {'-', '-', '-.'};

for i = 1:size(res,2)
    pt = semilogy(SNRdB, res(:,i)); hold on;
    pt.LineWidth = 1.5;
    
    pt.Marker = markers{i};
    pt.MarkerSize = 8;
    pt.Color = colors{i};   
    pt.LineStyle = lineStyles{i};
end

legend('ZF','ShaPeakg','ShaPeakh','Location','SouthWest')
xlabel('SNR (dB)')
ylabel('Bit Error Rate (BER)')
axis([min(SNRdB), max(SNRdB), 2e-4, 5e-1]) 
grid on

