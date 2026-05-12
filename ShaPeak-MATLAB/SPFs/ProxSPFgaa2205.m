function x = ProxSPFgaa2205(z,a,eta)
    % solving problem   
    % 
    % xi = argmin_t (t-zi)^2/2 + eta*g(t), 
    %          s.t. t\in[0,1] 
    % 
    % where g(t) = (t+a)^2/2-a^2/2    if t<=1/2 
    %       g(t) = (t-1-a)^2/2-a^2/2  if t>=1/2
    
    x  = zeros(size(z));
    if  eta < 0 
        disp('\eta must be a postive scalar !')
        return
    elseif eta > 0.5/a
        x(z >= 0.5 ) = 1;
    else     
        eta1 = 1+eta;

        ze   = z-a*eta;
        T    = find( (ze > 0) & (z <= 1/2) ); 
        x(T) = ze(T)/eta1;

        ze   = z+(1+a)*eta;
        T    = find( (z > 1/2) & (ze < eta1) ); 
        x(T) = ze(T)/eta1;
        x(ze>= eta1) = 1;
    end
end