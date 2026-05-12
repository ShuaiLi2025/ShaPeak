function out = funMIMO(x,key,data)
 % x \in R^{n x 1}
 % key \in {'f', 'g', 'H'}
 % data = (data.A,data.b), with data.A \in R^{m x n}, data.b \in R^{m x 1}
 
switch key
    case 'f'  
        obj  = @(var)norm(data.A*var-data.b,'fro')^2/2;          
        out  = obj(x);
    case 'g'
        grad = @(var)((data.A*var-data.b)'*data.A)';
        out  = grad(x); 
    case 'H' 
        if  length(x) <= 2e3
            out = data.AA; 
        else
            out = @(var)data.AA*var; 
        end
%         if  length(x) <= 6e3
%             out = data.A'*data.A; 
%         else
%             out = @(var)( (data.A*var)'*data.A )'; 
%         end
end       

end
