function x = ProxSPF111101(z,eta)
    % solving problem   
    % 
    % xi = argmin_t (t-zi)^2/2 + eta*g(t), 
    %          s.t. t\in[0,1] 
    % 
    % where g(t) = t if t<1 
    %       g(t) = 0 if t=1
    
    x  = zeros(size(z));
    if  eta < 0 
        disp('\eta must be a postive scalar !')
        return
    elseif eta > 0.5 
        x( z >=  0.5 ) = 1;
    else     
        ze    = z-eta;
        T     = find( ze > 0 ); 
        x(T)  = ze(T);
        x(ze >= 1-sqrt(2*eta)) = 1;
    end
end