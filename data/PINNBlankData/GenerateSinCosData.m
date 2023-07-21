clear;
clc;

addpath("..\SimpleCosSinTest\")

%% Generate data
numInternalCollocationPoints = 100;

x = linspace(0,2*pi,numInternalCollocationPoints^0.5);
t = linspace(0,2*pi,numInternalCollocationPoints^0.5);
[t, x] = meshgrid(t,x);
t = reshape(t,[1, numel(t)]);
x = reshape(x,[1, numel(x)]);

y = ones(1,numInternalCollocationPoints);

for index=1:numInternalCollocationPoints
    y(index) = solveU(x(index),t(index));
end

save SinCos.mat 'x' 't' 'y'
