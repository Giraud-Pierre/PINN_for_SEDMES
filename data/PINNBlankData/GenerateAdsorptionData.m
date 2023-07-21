clear;
clc;

addpath("..\AdsorptionExercise\")
%% Generate data
numInternalCollocationPoints = 100;

x = linspace(0,1,numInternalCollocationPoints^0.5);
t = linspace(0,1000,numInternalCollocationPoints^0.5);
[t, x] = meshgrid(t,x);
t = reshape(t,[1, numel(t)]);
x = reshape(x,[1, numel(x)]);

Cg = ones(1,numInternalCollocationPoints);
Cs = ones(1,numInternalCollocationPoints);

for index=1:numInternalCollocationPoints
    [Cg(index),Cs(index)] = solveAdsorptionPointByPoint(x(index),t(index));
end

%% Save Data
y = Cg;
save AdsorptionDataCg.mat 'x' 't' 'y'
y = Cs;
save AdsorptionDataCs.mat 'x' 't' 'y'
