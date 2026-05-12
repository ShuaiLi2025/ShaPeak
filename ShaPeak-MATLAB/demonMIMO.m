% demon classical MIMO problems
clc; close all; clear all; addpath(genpath(pwd));

M          = 1000;
N          = M;
scale      = sqrt(M);
Ntrials    = 3;
test       = 1; % I.I.D channels        if test =1
                % Correlated channels   if test =2 
                
SNR        = 0:2:10; % SNR range
sigma_snr  = sqrt( 2*N*10 .^ ( - (SNR) / 10 ) );

% QPSK constellation
mod_order  = 2;
P_per      = 1;
theta      = pi/(2^mod_order);

BER_ZF     = zeros(length(SNR),Ntrials);
BER_MMSE   = zeros(length(SNR),Ntrials);
BER_SPg    = zeros(length(SNR),Ntrials);
BER_SPh    = zeros(length(SNR),Ntrials);
BER_LB     = zeros(length(SNR),Ntrials);

Time_SPg    = zeros(length(SNR),Ntrials);
Time_SPh    = zeros(length(SNR),Ntrials);

wb = waitbar(0,'plez wait');
for i = 1:Ntrials
    fprintf('\n')
    display(['ntrials:' int2str(i)]);
    
    waitbar(i/Ntrials,wb);
    % generate channels
    H_c  = DataclassicalMIMO(M,N,test);
    H_c  = H_c/scale;
    H    = [real(H_c) -imag(H_c);
            imag(H_c) real(H_c)];
    
    % generating symbols
    Databits = round(rand(mod_order,N));
    symbol_index = bin2dec(char(Databits+48)');
    s_tr = sqrt(2)* pskmod(symbol_index,2^mod_order,theta);
    l_bit = symbol_decode(s_tr,mod_order,theta,'BER');
    
    % generating noise
    n_c  = (randn(M,1)+1i*randn(M,1))/scale;
    
    for j = 1:length(SNR)
            fprintf('\n SNR%4d \n',SNR(j));
            
            y_c = H_c*s_tr + (sigma_snr(j))*n_c;
            y   = [real(y_c);imag(y_c)];
                     
            %  ----- ZF -----------------
            x_zf          =  H\y;
            x_zf          = x_zf(1:N,:)+1i*x_zf(N+1:2*N,:);
            Bit_zf        = symbol_decode(x_zf,mod_order,theta,'BER');
            BER_ZF(j,i)   = length(find(Bit_zf-l_bit));
            
           % ------  MMSE --------------
            x_MMSE        = (H'*H+ (sigma_snr(j).^2)*eye(2*N)/(scale^2))\(H'*y);
            x_MMSE        = x_MMSE(1:N,:)+1i*x_MMSE(N+1:2*N,:);
            Bit_MMSE      = symbol_decode(x_MMSE,mod_order,theta,'BER');
            BER_MMSE(j,i) = length(find(Bit_MMSE-l_bit));            
            
           % ------  ShaPeakg --------------
            data.A        = 2*H;
            data.b        = H*ones(2*N,1)+y;
            data.AA       = data.A'*data.A;
            pars.preQ     = 1;
            pars.sigma    = 32/log10(N);
            pars.it0      = 10;
            pars.penrt    = 2;
            pen           = 1e-4*sqrt(N)*norm(data.b'*data.A,'Inf');
            
            func          = @(x,key)funMIMO(x,key,data); 
            pars.prob     = 'mimo';
            pars.penf     = 'gaa2205';
            out           = ShaPeakADMM(func,2*N,pen,pars);
            x_SPg         = 2*out.sol - 1;
            x_SPg         = x_SPg(1:N) + 1i*x_SPg(N+1:2*N);
            Bit_SPg       = symbol_decode(x_SPg,mod_order,theta,'BER');
            BER_SPg(j,i)  = length(find(Bit_SPg-l_bit));  
            Time_SPg(j,i) = out.time;
            
           % ------  ShaPeakh --------------
            pars.penf     = 'haa2205';
            out           = ShaPeakADMM(func,2*N,pen,pars);
            x_SPh         = 2*out.sol - 1;
            x_SPh         = x_SPh(1:N) + 1i*x_SPh(N+1:2*N);
            Bit_SPh       = symbol_decode(x_SPh,mod_order,theta,'BER');
            BER_SPh(j,i)  = length(find(Bit_SPh-l_bit)); 
            Time_SPh(j,i) = out.time;
            
           % ----- No-inteference lower bound ----
            x_LB          = NoInterference(H_c,s_tr,(sigma_snr(j))*n_c);
            Bit_LB        = symbol_decode(x_LB,mod_order,theta,'BER');
            BER_LB(j,i)   = length(find(Bit_LB-l_bit));
    end
    
end
close(wb)


% show BER results
%---- ZF ---------
BER_ZF_avg     = mean(BER_ZF,2)/(N*mod_order);
%---- MMSE ---------
BER_MMSE_avg   = mean(BER_MMSE,2)/(N*mod_order);
%---- shapeakg -------
BER_SPg_avg  = mean(BER_SPg,2)/(N*mod_order);
Time_SPg_avg = mean(Time_SPg,2);
%---- shapeakh -------
BER_SPh_avg  = mean(BER_SPh,2)/(N*mod_order);
Time_SPh_avg = mean(Time_SPh,2);
%---- no interference lower bound -------
BER_LB_avg   = mean(BER_LB,2)/(N*mod_order);


%--------- plot BER curve ----------------
figure('Renderer', 'painters','Position', [300 200 540 460])
axes('Position', [0.12 0.12 0.84 0.825])

res = [BER_LB_avg BER_ZF_avg BER_MMSE_avg ...
       BER_SPg_avg BER_SPh_avg]; 

colors = {'black', '#e9162d', '#ff8c00', '#f700f3', '#1fb819'};
markers = {'none', 'o', 'd', 's', '*'};
markerSizes = [8, 11, 7, 11, 7];
lineStyles = {'-', '-.', '-', '-', '-.'};

for i = 1:size(res,2)
    pt = semilogy(SNR, res(:,i)); hold on;
    pt.LineWidth = 1.5;
    pt.Color = colors{i};
    pt.LineStyle = lineStyles{i};

    if i > 1
        pt.Marker = markers{i};
        pt.MarkerSize = markerSizes(i);
    end
end

legend('LB','ZF','MMSE','ShaPeakg','ShaPeakh','Location','SouthWest')
xlabel('SNR (dB)')
ylabel('Bit Error Rate (BER)')
axis([min(SNR), max(SNR), 3e-4, 1]) 
grid on



