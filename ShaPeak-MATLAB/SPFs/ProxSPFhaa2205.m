function x = ProxSPFhaa2205(z,a,eta)
    % solving problem   
    % 
    % xi = argmin_t (t-zi)^2/2 + eta*g(t), 
    %          s.t. t\in[0,1] 
    % 
    % where g(t) = a^2/2-(t-a)^2/2    if t<=1/2 
    %       g(t) = a^2/2-(t+a-1)^2/2  if t>=1/2
    
    x  = zeros(size(z));
    if  eta < 0 
        disp('\eta must be a postive scalar !')
        return
    elseif eta > 0.5/a
        x( z  >= 0.5 ) = 1;
    else     
        eta1 = 1-eta;

        tmp  = z-a*eta;
        T    = find( (tmp > 0) & (z <= 1/2) ); 
        x(T) = tmp(T)/eta1;

        tmp  = z+(a-1)*eta;
        T    = find( (z > 1/2) & (tmp < eta1) ); 
        x(T) = tmp(T)/eta1;
        x(tmp>= eta1) = 1;
    end
end