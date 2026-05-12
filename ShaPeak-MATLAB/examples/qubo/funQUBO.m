function out = funQUBO(x,key,MatQ)
 % x    \in R^{n x 1}
 % key  \in {'f', 'g', 'H'}
 % MatQ \in R^{m x n} is a symmetric matrix
 
    switch key
        case 'f'  
            xQx  = @(var)var'*MatQ*var; 
            out  = -0.5*xQx(x);
        case 'g' 
            Qx   = @(var)MatQ*var;
            out  = -Qx(x);
        case 'H' 
            out  = 0;  
    end         
end