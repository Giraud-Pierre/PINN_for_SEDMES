function [Cg,Cs] = solveAdsorption(x_solve, T_solve)
% Function to solve analytically the PDEs
ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;

Cg = zeros(length(x_solve),length(T_solve));
Cs = zeros(length(x_solve),length(T_solve));

for Tindex = 1:length(T_solve)
    for Xindex = 1:length(x_solve)
        tau = kg*as/((1-epsb)*Ke)*(T_solve(Tindex) - x_solve(Xindex)/ug);
        xi = kg*as/(epsb*ug)*x_solve(Xindex);
        fnc = @(u) exp(-u).*besseli(0, sqrt(4*tau*u), 1)./exp(-abs(real(sqrt(4*tau*u))));
        I = integral(fnc, 0 , xi);
        e = exp(-tau);
        g = besseli(0, sqrt(4*tau*xi))*exp(-xi);
        Cg(Xindex,Tindex) = real(1 - e*I);
        Cs(Xindex,Tindex) = real(Ke*(1 - e*(I + g)));
    end
end

end
