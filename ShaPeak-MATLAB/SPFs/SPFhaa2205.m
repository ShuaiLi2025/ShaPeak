function g = SPFhaa2205(x,a)
         g = sum( a^2-(x(x<=0.5)-a).^2);
         g = (g + sum( a^2-(x(x>0.5)+(a-1)).^2 ) )/2;
end