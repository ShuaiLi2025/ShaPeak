function [H_tilde,y_tilde,Homega,v_symbol,theta,H_pinv]= data1bMIMO(M, N, SNRdB, mod_order)
    
    
    SNR        = 10.^(SNRdB/10);
    sigma_C    = 2*N./SNR;
    sigma_real = sigma_C/2;
    sigma_OE   = sqrt(sigma_real)+0.5; % over-estimated noise standard variance

    % QPSK constellation
    theta      = pi/(2^mod_order);

    % generate channels
    scale      = sqrt(2);
    H          = (randn(M,N)+1i*randn(M,N))/scale;
    H_pinv     = pinv(H);

    % generating symbols
    Databits     = round(rand(mod_order,N));
    symbol_index = bin2dec(char(Databits+48)');
    symbol_mat   = sqrt(2)* pskmod(symbol_index,2^mod_order,theta);
    v_symbol     = symbol_decode(symbol_mat,mod_order,theta,'BER');

    % generating noise
    n_ch      = (randn(M,1)+1i*randn(M,1))/scale;
    nf        = sqrt(sigma_C)*n_ch;
    y         = H*symbol_mat+nf;
    y_r       = [real(y);imag(y)]; 
    H_tilde   = [real(H), -imag(H);imag(H), real(H)];
    y_tilde   = sign(y_r);
    G_matrix  = (diag(y_tilde)*H_tilde).';
    Homega    = G_matrix'/sigma_OE; 
end