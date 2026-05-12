function g = SPFgaa2205(x,a)
         g = sum( (x(x<=0.5)+a).^2-a^2 );
         g = (g + sum( (x(x>0.5)-(1+a)).^2-a^2 ) )/2;
end