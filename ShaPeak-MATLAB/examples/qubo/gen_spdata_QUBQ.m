function data = gen_spdata_QUBQ(nvar, neg_ratio, dens_ratio)
    rowIdx = [];
    colIdx = [];
    vals   = [];
    
    for i = 1:nvar
      
        n_possible = nvar - i + 1;
        mask = rand(1, n_possible) < dens_ratio;
        js = find(mask) + i - 1;  
        
        if ~isempty(js)
            k = numel(js);
            rvals = randi([10, 100], 1, k);
            signs = 2 * (rand(1, k) > neg_ratio) - 1;
            rvals = rvals .* signs;
            
            rowIdx = [rowIdx, repmat(i, 1, k)];
            colIdx = [colIdx, js];
            vals   = [vals, rvals];
        end
    end
    
    U = sparse(rowIdx, colIdx, vals, nvar, nvar);
    data = U + U' - spdiags(diag(U), 0, nvar, nvar);
end
