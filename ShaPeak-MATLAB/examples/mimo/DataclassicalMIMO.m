function  H_c = DataclassicalMIMO(M,N,test)
    disp(' Data is generating ...')
    % generate channels
    switch test
        case 1 % I.I.D channels
            H_c = (randn(M,N)+1i*randn(M,N));
        case 2 % Correlated channels
            r   = 0.2;
            Rr  = zeros(M); % receive channel coherence matrix
            for i = 1:M
                for j = 1:M
                    Rr(i,j) = r^(abs(i-j));
                end
            end
            Rr_sqrt = sqrtm(Rr);

            Rt = zeros(N); % transmit channel coherence matrix
            for i = 1:N
                for j = 1:N
                    Rt(i,j) = r^(abs(i-j));
                end
            end
            Rt_sqrt = sqrtm(Rt);
            H_c     = Rr_sqrt* (randn(M,N)+1i*randn(M,N))*Rt_sqrt;      
    end
    disp(' Done data generation !!!')
end