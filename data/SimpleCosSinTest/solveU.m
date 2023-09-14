function [u] = solveU(x,t)
u = zeros(1,numel(x));

for i =1:numel(x)
u(i) = sin(3 * x(i)) * cos(2 * t(i));
% u(i) = sin(3 * x(i) * 2 * t(i));
end
end