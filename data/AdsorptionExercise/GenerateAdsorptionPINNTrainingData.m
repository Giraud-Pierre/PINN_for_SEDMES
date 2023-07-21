clear; 
clc;
format long e

%% Constants
L = 1.0;
Cin = 1.0;
Ke = 10;
tend = 1000;

%% Boundary conditions
numBoundaryConditionPoints = [200 200];

xBC1 = zeros(1,numBoundaryConditionPoints(1));
tBC1 = linspace(0,tend,numBoundaryConditionPoints(1));
CgBC1 = ones(1,numBoundaryConditionPoints(1)) * Cin;

xBC2 = ones(1,numBoundaryConditionPoints(2));
tBC2 = linspace(0,tend,numBoundaryConditionPoints(2));
CgBC2 = zeros(1,numBoundaryConditionPoints(2));

%% Initial conditions
numInitialConditionPoints  = 200;

xIC = linspace(0,L,numInitialConditionPoints);
tIC = zeros(1,numInitialConditionPoints);
CgIC = zeros(1,numInitialConditionPoints); 
CsIC = zeros(1,numInitialConditionPoints);

%% Collocation points for PDEs
numInternalCollocationPoints = 1000;

points = rand(2,numInternalCollocationPoints);

x = points(1,:) * L;
t = points(2,:) * tend;

%% Data points to see if it improves the model
NumdataPointsOnT = 6;
NumdataPointsOnX = 20;

xd = linspace(L/(3*(NumdataPointsOnX*2)),(NumdataPointsOnX*2 - 1) * L/(3*(NumdataPointsOnX*2)),NumdataPointsOnX);
td = linspace(0,tend,NumdataPointsOnT);
[td,xd] = meshgrid(td,xd);

xd = reshape(xd,[1,numel(xd)]);
td = reshape(td,[1,numel(td)]);

[Cgd, Csd] = solveAdsorptionPointByPoint(xd, td);

%% Save the data

save("TrainingDataPINNWithDataPoints.mat", 'xBC1', 'tBC1', 'CgBC1', 'xBC2', ...
    'tBC2', 'CgBC2', 'xIC', 'tIC', 'CgIC', 'CsIC', 'x', 't', 'xd', 'td', 'Cgd', 'Csd')