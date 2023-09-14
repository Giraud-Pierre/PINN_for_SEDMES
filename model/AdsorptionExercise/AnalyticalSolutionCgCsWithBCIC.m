clear; 
clc;
format long e

addpath("..\..\data\AdsorptionExercise\")
addpath("..\..\losses\AdsorptionLosses\AnalyticalSolutions\")

%% Constant
L = 1.0;
Cin = 1.0;
Ke = 10;
tend = 1000;

%% Get data for initial condition and collocation points
%Boundary conditions 1
numBoundaryConditionPoints = [25 25];

xBC1 = zeros(1,numBoundaryConditionPoints(1));
tBC1 = linspace(2*tend/10,tend,numBoundaryConditionPoints(1));
%not starting at t = 0 to not conflict with initial condition

CgBC1 = Cin * ones(1,numBoundaryConditionPoints(1));
% No boundary condition 1 on Cs

%Initial conditions
numInitialConditionPoints  = 50;

xIC = linspace(0,L,numInitialConditionPoints);
tIC = zeros(1,numInitialConditionPoints);
CgIC = zeros(1,numInitialConditionPoints);
CsIC = zeros(1,numInitialConditionPoints);

%Computing Data Point

% numInternalCollocationPoints = 200;
% 
% points = rand(numInternalCollocationPoints,2);
% 
% data_X = points(:,1) * L;
% data_T = points(:,2) * tend;

X_test = linspace(2*L/10,L,100);
t_test = linspace(2*tend/10,tend,5);
[t_mesh, X_mesh] = meshgrid(t_test,X_test);
data_X = reshape(X_mesh,[1,numel(X_mesh)]);
data_T = reshape(t_mesh,[1,numel(t_mesh)]);

[data_Cg, data_Cs] = solveAdsorptionPointByPoint(data_X, data_T);


%% Test SolveAdsorption (analytical solution)
% figure()
% 
% subplot(2,2,1)
% scatter(data_X, data_Cg,10,data_T)
% subplot(2,2,2)
% scatter(data_X, data_Cs,10,data_T)
% subplot(2,2,3)
% scatter(data_T, data_Cg,10,data_X)
% subplot(2,2,4)
% scatter(data_T, data_Cs,10,data_X)


%% Creating dlarray for deeplearning
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    XICdl = gpuArray(dlarray(xIC,"CB"));
    TICdl = gpuArray(dlarray(tIC,"CB"));
    CgICdl = gpuArray(dlarray(CgIC,"CB"));
    CsICdl = gpuArray(dlarray(CsIC, "CB"));
    XBC1dl = gpuArray(dlarray(xBC1, "CB"));
    TBC1dl = gpuArray(dlarray(tBC1, "CB"));
    CgBC1dl = gpuArray(dlarray(CgBC1, "CB"));
    Xdl = gpuArray(dlarray(data_X,"CB"));
    Tdl = gpuArray(dlarray(data_T,"CB"));
    Cgdl = gpuArray(dlarray(data_Cg,"CB"));
    Csdl = gpuArray(dlarray(data_Cs,"CB"));
    "GPU"
else
    XICdl = dlarray(xIC,"CB");
    TICdl = dlarray(tIC,"CB");
    CgICdl = dlarray(CgIC,"CB");
    CsICdl = dlarray(CsIC, "CB");
    XBC1dl = dlarray(xBC1, "CB");
    TBC1dl = dlarray(tBC1, "CB");
    CgCB1dl = dlarray(CgBC1, "CB");
    Xdl = dlarray(data_X,"CB");
    Tdl = dlarray(data_T,"CB");
    Cgdl = dlarray(data_Cg,"CB");
    Csdl = dlarray(data_Cs,"CB");
end

%% Test dlArray
% figure()
% subplot(2,2,1)
% scatter(extractdata(Xdl), extractdata(Cgdl), 10, extractdata(Tdl))
% subplot(2,2,2)
% scatter(extractdata(Xdl), extractdata(Csdl), 10, extractdata(Tdl))
% subplot(2,2,3)
% scatter(extractdata(Tdl), extractdata(Cgdl), 10, extractdata(Xdl))
% subplot(2,2,4)
% scatter(extractdata(Tdl), extractdata(Csdl), 10, extractdata(Xdl))

%% Neural network architecture for Cg
numLayers = 6;
numNeurons = 64;

layersCg = featureInputLayer(2);

for i = 1:numLayers-1
    layersCg = [
        layersCg
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layersCg = [
    layersCg
    fullyConnectedLayer(1)]

netCg = dlnetwork(layersCg) %convert the network into a dlnetwork object

%% Neural network architecture for Cs
numLayers = 6;
numNeurons = 64;

layersCs = featureInputLayer(2);

for i = 1:numLayers-1
    layersCs = [
        layersCs
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layersCs = [
    layersCs
    fullyConnectedLayer(1)]

netCs = dlnetwork(layersCs) %convert the network into a dlnetwork object


%% Test net

% concat = cat(1,Xdl,Tdl);
% 
% Cg_test = forward(netCg,concat);
% Cs_test = forward(netCs, concat);

%% Training with LBFGS
numEpochs = 300;
solverStateCg = lbfgsState;
solverStateCs = lbfgsState;

lossCg = @(net) dlfeval(@LossCgAnalyticalWithBCIC,net, XICdl, TICdl, CgICdl, XBC1dl, TBC1dl, CgBC1dl, Xdl,Tdl,Cgdl);
lossCs = @(net) dlfeval(@LossCsAnalyticalWithBCIC,net,XICdl,TICdl,CsICdl,Xdl,Tdl,Csdl);

monitorCg = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

monitorCs = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [netCg, solverStateCg] = lbfgsupdate(netCg,lossCg,solverStateCg);

    updateInfo(monitorCg,Epoch=i);
    recordMetrics(monitorCg,i,TrainingLoss=solverStateCg.Loss);
end

for i = 1:numEpochs
    [netCs, solverStateCs] = lbfgsupdate(netCs,lossCs,solverStateCs);

    updateInfo(monitorCs,Epoch=i);
    recordMetrics(monitorCs,i,TrainingLoss=solverStateCs.Loss);
end

%% Training with Adam
% numEpochs = 300;
% initial_learning_rate = 0.01;
% averageGradCg = [];
% averageSqGradCg = [];
% averageGradCs = [];
% averageSqGradCs = [];
% 
% lossFcnCg = @(net) dlfeval(@LossCgAnalyticalWithBCIC,net, XICdl, TICdl, CgICdl, XBC1dl, TBC1dl, CgBC1dl, Xdl,Tdl,Cgdl);
% lossFcnCs = @(net) dlfeval(@LossCsAnalyticalWithBCIC,net,XICdl,TICdl,CsICdl,Xdl,Tdl,Csdl);
% 
% monitorCg = trainingProgressMonitor( ...
%     Metrics="TrainingLoss", ...
%     Info="Epoch", ...
%     XLabel="Epoch");
% 
% monitorCs = trainingProgressMonitor( ...
%     Metrics="TrainingLoss", ...
%     Info="Epoch", ...
%     XLabel="Epoch");
% 
% learning_rate = initial_learning_rate;
% for i = 1:numEpochs
%     [loss, gradientsCg] = lossFcnCg(netCg);
% 
%     [netCg, averageGradCg, averageSqGradCg] = adamupdate(netCg,gradientsCg,averageGradCg, averageSqGradCg, i, learning_rate);
% 
%     updateInfo(monitorCg,Epoch=i);
%     recordMetrics(monitorCg,i,TrainingLoss=loss);
% end
% 
% learning_rate = initial_learning_rate;
% for i = 1:numEpochs
%     [loss, gradientsCs] = lossFcnCs(netCs);
% 
%     [netCs, averageGradCs, averageSqGradCs] = adamupdate(netCs,gradientsCs,averageGradCs, averageSqGradCs, i, learning_rate);
% 
%     updateInfo(monitorCs,Epoch=i);
%     recordMetrics(monitorCs,i,TrainingLoss=loss);
% end

%% testing and ploting with same data as training
xplot = linspace(0,1,100);
tplot = linspace(0,1000,6);
[tplot,xplot] = meshgrid(tplot,xplot);
tplot = reshape(tplot,[1,numel(tplot)]);
xplot = reshape(xplot,[1,numel(xplot)]);
tdlplot = dlarray(tplot,"CB");
xdlplot = dlarray(xplot,"CB");

Cg_pred = forward(netCg, cat(1,xdlplot,tdlplot));
Cs_pred = forward(netCs, cat(1,xdlplot,tdlplot));
[CgExact, CsExact] = solveAdsorptionPointByPoint(xplot,tplot);

errCg = norm(extractdata(Cg_pred) - CgExact) / norm(CgExact)
errCs = norm(extractdata(Cs_pred) - CsExact) / norm(CsExact)

figure('Name',sprintf('Analytical solution for Cs and Cg together, errorCg = %f and errorCs  =%f',errCg, errCs))
subplot(2,2,1)
plot3(xplot,tplot,extractdata(Cg_pred),'*','DisplayName',"Cg predicted")
legend()
subplot(2,2,2)
plot3(xplot,tplot,CgExact,'*', 'DisplayName', "exact Cg")
legend()
subplot(2,2,3)
plot3(xplot,tplot,extractdata(Cs_pred),'*','DisplayName',"Cs predicted")
legend()
subplot(2,2,4)
plot3(xplot,tplot,CsExact,'*', 'DisplayName', "exact Cs")
legend()

%% Saving result

% savefig('../../results/AdsorptionExercise/AnalyticalSolutions/AnalyticalSolutionCgCsWithBCIC.fig')
% save('../../results/AdsorptionExercise/AnalyticalSolutions/AnalyticalSolutionCgCsWithBCIC',"xplot","tplot","CgExact","Cg_pred","CsExact","Cs_pred","netCg","netCs")