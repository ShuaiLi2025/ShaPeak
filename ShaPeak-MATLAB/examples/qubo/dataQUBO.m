function data = dataQUBO(nvar, neg_ratio, dens_ratio)
   
    data = randi([10, 100], nvar, nvar);
    
    neg = rand(nvar, nvar);
    neg = (neg > neg_ratio);
    neg = neg * 2 - 1; 
    
    dens = rand(nvar, nvar);
    dens = (dens < dens_ratio);
    
    data = data .* neg .* dens; 
    
    data = triu(data); 
    data = data + data'; 
    data(1:(nvar+1):end) = data(1:(nvar+1):end)/ 2;
    
end