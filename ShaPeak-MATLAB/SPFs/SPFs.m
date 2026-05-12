function g = SPFs(x,u,pars,key)
a = pars(1);
b = pars(2);
p = pars(3);
q = pars(4);
       
T1 = find(x<u); 
T2 = find(x==u); 
T3 = find(x>u);
g  = 0;
switch key
    case 'g'
        if ~isempty(T1)
            g = g + (norm(x(T1)+a,p)^p - length(T1)*a^p)/p; 
        end
        if ~isempty(T3)
            g = g +  (norm(1+b-x(T3),q)^q - length(T3)*b^q)/q; 
        end
        if ~isempty(T2)
            tmp = min( ((u+a)^p-a^p)/p,((1+b-u)^q-b^q)/q ); 
            g   = g + length(T2)*tmp;
        end
    case 'h'
        if ~isempty(T1)
            g = g + (length(T1)*a^p - norm(x(T1)-a,p)^p)/p; 
        end
        if ~isempty(T3)
            g = g + (length(T3)*b^q - norm(b-1+x(T3),q)^q)/q; 
        end
        if ~isempty(T2)
            tmp = min( (a^p-(u-a)^p)/p,(b^q-(b-1+u)^q)/q ); 
            g   = g + length(T2)*tmp;
        end
end
