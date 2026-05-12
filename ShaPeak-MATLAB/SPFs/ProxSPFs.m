function prox = ProxSPFs(z,u,eta,pars,key)
a = pars(1);
b = pars(2);
p = pars(3);
q = pars(4);      

switch key
    case 'g'
        if  p==1
            t1 = max(0,min(u,z-eta));          
        else
            t1 = max(0,min(u,(z-a*eta)/(1+eta)));
        end        
        f1 = (t1-z).^2/2 + (eta/p)*( (t1+a).^p - a^p);  
        
        if  q==1
            t2 = max(u,min(1,z+eta));
        else
            t2 = max(u,min(1,(z+(1+b)*eta)/(1+eta)));
        end
        f2 = (t2-z).^2/2 + (eta/q)*( (1+b-t2).^q - b^q);  
    case 'h'
        if  p==1
            t1 = max(0,min(u,z-eta));     
        else
            if  eta<1
                t1 = max(0,min(u,(z-a*eta)/(1-eta)));         
            elseif eta==1
                t1 = zeros(size(z));
                t1(z>a) = u;  
            else
                t1 = zeros(size(z));
            end
        end        
        f1 = (t1-z).^2/2 + (eta/p)*( a^p-(a-t1).^p ); 
        
        if  q==1
            t2 = max(u,min(1,z+eta));
        else
            if  eta<1
                t2 = max(u,min( 1, (z+(b-1)*eta)/(1-eta) ));
            elseif eta==1
                t2 = ones(size(z));
                t2(z<1-b) = u; 
            else
                t2 = ones(size(z));
            end
        end 
        f2 = (t2-z).^2/2 + (eta/q)*( b^q-(t2-1+b).^q ); 
end

[~,T] = min([f1 f2],[],2);  
T     = T+(0:1:(length(z)-1))'*2;  
tt    = [t1'; t2'];
prox  = tt(T);  %prox'
end
