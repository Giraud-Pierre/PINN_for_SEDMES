clear; 
clc;
format long e

%% Data
%Boundary conditions
numBoundaryConditionPoints = [25 25];

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

%Initial conditions
numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

%Colocation points
numInternalCollocationPoints = 100;

points = rand(numInternalCollocationPoints,2);

dataX = 2*points(:,1)-1;
dataT = points(:,2);


DataU = ones(1,numInternalCollocationPoints);

for index=1:numInternalCollocationPoints
    DataU(index) = solveBurgers(dataX(index),dataT(index),0.01/pi);
end

%% Network
numLayers = 9;
numNeurons = 20;

parameters = struct;

layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(1)]

net = dlnetwork(layers)
%%
X = dlarray(dataX,"BC");
T = dlarray(dataT,"BC");
U = dlarray(DataU,"CB");
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0,"CB");

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    X = gpuArray(X);
    T = gpuArray(T);
    U = gpuArray(U);
    X0 = gpuArray(X0);
    T0 = gpuArray(T0);
    U0 = gpuArray(U0);
    "GPU"
end

%% Training
solverState = lbfgsState;
numEpochs = 500;

lossFcn = @(net) dlfeval(@modelLoss,net,X,T,U,X0,T0,U0);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end

%% Testing and ploting
tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);
XTest = dlarray(XTest,"CB");

figure
tiledlayout("flow")

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);
    TTest = dlarray(TTest,"CB");

    % Make predictions.
    XTTest = cat(1,XTest,TTest);
    UPred = forward(net,XTTest);

    % Calculate target.
    UTest = solveBurgers(extractdata(XTest),t,0.01/pi);

    % Calculate error.
    UPred = extractdata(UPred);
    err = norm(UPred - UTest) / norm(UTest);

    % Plot prediction.
    nexttile
    plot(XTest,UPred,"-",LineWidth=2);
    ylim([-1.1, 1.1])

    % Plot target.
    hold on
    plot(XTest, UTest,"--",LineWidth=2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

legend(["Prediction" "Target"])

%% SOlve burgers
function U = solveBurgers(X,t,nu)

% Define functions.
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));

% Initialize solutions.
U = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);

    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
    if abs(x) ~= 1
        fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
        uxt = -integral(fun,-inf,inf);
        fun = @(eta) f(x-eta) .* g(eta);
        U(i) = uxt / integral(fun,-inf,inf);
    end
end

end

%% Loss functions
function [loss,gradients] = modelLoss(net,X,T,U,X0,T0,U0)

% Make predictions with the initial conditions.
XT = cat(1,X,T);
U_pred = forward(net,XT);

mseF = l2loss(U_pred,U);

% Calculate mseU. Enforce initial and boundary conditions.
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
loss = mseF + mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end