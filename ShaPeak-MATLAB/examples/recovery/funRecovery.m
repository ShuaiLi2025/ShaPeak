function  out  = funRecovery(x,key,p,data)
 % x \in R^{n x 1}
 % key \in {'f', 'g', 'H'}
 % p>1
 % data = (data.A,data.b), with data.A \in R^{m x n}, data.b \in R^{m x 1}
 
    n   = length(x);
    if ~isequal(key,'H') || (isequal(key,'H') && p~=2)
        Ax  = @(var)(data.A*var);          
        Axb = Ax(x)-data.b; 
    end
    
    if ~isequal(key,'f') || ~(isequal(key,'H') && n <= 1e3)
        Atx  = @(var)(var'*data.A)';          
    end
    
    switch key
        case 'f'  
            out = norm(Axb,p)^p/p;
        case 'g'
            if  p   ==2
                out = Atx( Axb ); 
            else
                out = Atx( sign(Axb).*abs(Axb).^(p-1) );
            end
        case 'H' 
            if  p ==2    
                if  n  <= 1e3
                    out = data.A'*data.A; 
                else
                    out = @(var)( Atx(data.A*var) ); 
                end
            else 
                tmp = (p+(p>2))*(abs(Axb).^(p-2)+1e-8*(p<2));
                out = @(var)Atx( tmp.*Ax(var) );
            end
    end     

end
% 
% end
