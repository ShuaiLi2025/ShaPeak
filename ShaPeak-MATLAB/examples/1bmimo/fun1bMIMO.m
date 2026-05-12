function out = fun1bMIMO(x,key,Hy,HH)

    
    fHy         = @(var)Hy*var;
    Hyx         = fHy(2*x-1);   
    Phi         = normcdf(Hyx);
    Phi(Phi==0) = 1e-200;  
    
    switch key
        case 'f'; out  = -sum(log(Phi));
        case 'g'; fHyt = @(var)(var'*Hy)';
                  out  = -sqrt(2/pi)*fHyt(exp(-0.5*Hyx.^2)./Phi);
        case 'H'
%             tmp  = exp(-0.5*Hyx.^2)./Phi/sqrt(2*pi);
%             if   length(x) < 2e3
%                  out = Hy'*((4*tmp.*(Hyx + tmp)).*Hy); 
%             else
%                  out  = @(v)(( (4*tmp.*(Hyx + tmp)).*fHy(v) )'*Hy)'; 
%             end

%                   out  = @(v)(( fHy(4*v) )'*Hy)';
                  
%                   HH   = Hy'*Hy;
                  out  = @(var)HH*var;
                  
    end     

end
