function [Cg,Cs] = solveAdsorptionPointByPoint(x, t)
if size(x) ~= size(t)
    error("Size of x = " + num2str(size(x)) + " and size of t = " + num2str(size(t)) + ...
        ". x and t should be of the same size to solve adsorption. They " + ...
        "are the coordinates in each dimension of the points you want to " + ...
        "solve the adsorption on. If it is not the case, you can try " + ...
        "using meshgrid to make it so.")
end

ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;

X = reshape(x,[1,numel(x)]);
T = reshape(t,[1,numel(t)]);

Cg = zeros(1,numel(x));
Cs = zeros(1,numel(x));

for index = 1:numel(x)
    tau = kg*as/((1-epsb)*Ke)*(T(index) - X(index)/ug);
    xi = kg*as/(epsb*ug)*X(index);
    fnc = @(u) exp(-u).*besseli(0, sqrt(4*tau*u), 1)./exp(-abs(real(sqrt(4*tau*u))));
    I = integral(fnc, 0 , xi);
    e = exp(-tau);
    g = besseli(0, sqrt(4*tau*xi))*exp(-xi);
    Cg(index) = real(1 - e*I);
    Cs(index) = real(Ke*(1 - e*(I + g)));
end
Cg = reshape(Cg, size(x));
Cs = reshape(Cs, size(x));

end