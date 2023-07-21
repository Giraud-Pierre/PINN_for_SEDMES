clear;
clc;

addpath("..\BurgersEquation\")

%% Generate data
numInternalCollocationPoints = 100;

x = linspace(-1,1,numInternalCollocationPoints^0.5);
t = linspace(0.05,1,numInternalCollocationPoints^0.5);
[t, x] = meshgrid(t,x);
t = reshape(t,[1, numel(t)]);
x = reshape(x,[1, numel(x)]);

y = ones(1,numInternalCollocationPoints);

for index=1:numInternalCollocationPoints
    y(index) = solveBurgers(x(index),t(index),0.01/pi);
end

save BurgerData.mat 'x' 't' 'y'