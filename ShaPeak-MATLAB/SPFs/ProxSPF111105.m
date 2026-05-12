function x = ProxSPF111105(z,eta)
    % solving problem   
    % 
    % xi = argmin_t (t-zi)^2/2 + eta*g(t), 
    %          s.t. t\in[0,1] 
    % 
    % where g(t) = t    if t<=1/2 
    %       g(t) = 1-t  if t>=1/2
    
    x  = zeros(size(z));
    if  eta < 0 
        disp('\eta must be a postive scalar !')
        return
    elseif eta > 0.5 
        x( z >= 0.5 ) = 1;
    else     
        T     = find( (z > eta) & (z <= 1/2) ); 
        x(T)  = z(T) - eta;
        ze    = z+eta;
        T     = find( (z > 1/2) & (ze < 1) ); 
        x(T)  = ze(T);
        x(ze >= 1) = 1;
    end
  
end