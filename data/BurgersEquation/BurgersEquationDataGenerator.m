%% Boundary conditions
numBoundaryConditionPoints = [25 25];

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

%% Initial conditions
numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

%% Colocation points for PINN
numInternalCollocationPoints = 10000;

points = rand(2,numInternalCollocationPoints);

dataXPINN = 2*points(1,:)-1;
dataTPINN = points(2,:);

%% Colocation points for analytical solution
numInternalCollocationPoints = 100;

points = rand(2,numInternalCollocationPoints);

dataXAnaSol = 2*points(1,:)-1;
dataTAnaSol = points(2,:);


DataUAnaSol = ones(1,numInternalCollocationPoints);

for index=1:numInternalCollocationPoints
    DataUAnaSol(1,index) = solveBurgers(dataXAnaSol(1,index),dataTAnaSol(1,index),0.01/pi);
end

%% Save workspace
save("BurgersEquationTrainingData.mat","X0","T0","U0", ...
    "dataXPINN", "dataTPINN","dataXAnaSol","dataTAnaSol", ...
    "DataUAnaSol")