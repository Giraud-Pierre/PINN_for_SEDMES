clear; 
clc;
format long e

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

X_test = linspace(0,L,100);
t_test = linspace(0,tend,6);
[t_mesh, X_mesh] = meshgrid(t_test,X_test);
data_X = reshape(X_mesh,[1,numel(X_mesh)]);
data_T = reshape(t_mesh,[1,numel(t_mesh)]);

[data_Cg, data_Cs] = solveAdsorptionRandom(data_X, data_T);


%% Test SolveAdsorption (analytical solution)
figure()

subplot(2,2,1)
scatter(data_X, data_Cg,10,data_T)
subplot(2,2,2)
scatter(data_X, data_Cs,10,data_T)
subplot(2,2,3)
scatter(data_T, data_Cg,10,data_X)
subplot(2,2,4)
scatter(data_T, data_Cs,10,data_X)


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
figure()
subplot(2,2,1)
scatter(extractdata(Xdl), extractdata(Cgdl), 10, extractdata(Tdl))
subplot(2,2,2)
scatter(extractdata(Xdl), extractdata(Csdl), 10, extractdata(Tdl))
subplot(2,2,3)
scatter(extractdata(Tdl), extractdata(Cgdl), 10, extractdata(Xdl))
subplot(2,2,4)
scatter(extractdata(Tdl), extractdata(Csdl), 10, extractdata(Xdl))

%% Neural network architecture for Cg
numLayers = 9;
numNeurons = 20;

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
numLayers = 9;
numNeurons = 20;

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

concat = cat(1,Xdl,Tdl);

Cg_test = forward(netCg,concat);
Cs_test = forward(netCs, concat);

%% Training with LBFGS
% numEpochs = 1000;
% solverStateCg = lbfgsState;
% solverStateCs = lbfgsState;
% 
% lossCg = @(net) dlfeval(@LossCgAnaSol,net, XICdl, TICdl, CgICdl, XBC1dl, TBC1dl, CgBC1dl, Xdl,Tdl,Cgdl);
% 
% monitorCg = trainingProgressMonitor( ...
%     Metrics="TrainingLoss", ...
%     Info="Epoch", ...
%     XLabel="Epoch");
% 
% % monitorCs = trainingProgressMonitor( ...
% %     Metrics="TrainingLoss", ...
% %     Info="Epoch", ...
% %     XLabel="Epoch");
% 
% for i = 1:numEpochs
%     [netCg, solverStateCg] = lbfgsupdate(netCg,lossCg,solverStateCg);
% 
%     updateInfo(monitorCg,Epoch=i);
%     recordMetrics(monitorCg,i,TrainingLoss=solverStateCg.Loss);
% end
% 
% % for i = 1:numEpochs
% %     [netCs, solverStateCs] = lbfgsupdate(netCs,lossCs,solverStateCs);
% % 
% %     updateInfo(monitorCs,Epoch=i);
% %     recordMetrics(monitorCs,i,TrainingLoss=solverStateCs.Loss);
% % end

%% Training with Adam
numEpochs = 300;
initial_learning_rate = 0.01;
averageGrad = [];
averageSqGrad = [];

lossFcn = @(net) dlfeval(@LossCgAnaSol,net, XICdl, TICdl, CgICdl, XBC1dl, TBC1dl, CgBC1dl, Xdl,Tdl,Cgdl);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

learning_rate = initial_learning_rate;
for i = 1:numEpochs
    [loss, gradients] = lossFcn(netCg);

    [netCg, averageGrad, averageSqGrad] = adamupdate(netCg,gradients,averageGrad, averageSqGrad, i, learning_rate);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=loss);
end
%% testing and ploting with same data as training
X_test = linspace(0,L,100);
t_test = linspace(0,tend,6);

figure()
for t_idx=1:numel(t_test)
    T = ones(1,numel(X_test)) * t_test(t_idx);
    XT = cat(1,X_test, T);

    Cg_pred = forward(netCg, dlarray(XT,"CB"));
    %Cs_pred = forward(netCs, XT_test);

    [Cg_exact, Cs_exact] = solveAdsorptionRandom(X_test,T);

    subplot(2,2,1), hold on
    plot(X_test,Cg_pred)
    %subplot(2,1,2), hold on
    %plot(X_test,Cs_pred)
    subplot(2,2,3), hold on
    plot(X_test,Cg_exact)
    %subplot(2,2,4), hold on
    %plot(X_test,Cs_exact)
end
