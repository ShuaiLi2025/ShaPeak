function g = SPFhaa2200(x,a)
         g = sum(a^2-(x(x>0)+(a-1)).^2)/2;
end
