function g = SPFgaa2200(x,a)
         g = sum((x(x>0)-(1+a)).^2-a^2)/2;
end
