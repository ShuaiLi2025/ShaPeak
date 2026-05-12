function g = SPFgaa2201(x,a)
         g = sum((x(x<1)+a).^2-a^2)/2;
end
