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

%% Solving Cg Cs analytically
%Computing Data Point
NumdataPointsOnT = 6;
NumdataPointsOnX = 20;

xData2D = linspace(0,L,NumdataPointsOnX);
tData2D = linspace(0,tend,NumdataPointsOnT);

[CgData2D, CsData2D] = solveAdsorption(xData2D,tData2D);
%Compute the analytical solutions of Cg and Cs and return 2 arrays of shape
%NumdataPointsOnX * NumdataPointsOnT which give the concentration function
%of time and space (CgData2D(x,t) is Cg at time tData2D(t) and space xData2D(x))

%% Test SolveAdsorption (analytical solution)
figure('Name','Test SolveAdsorption solver')

for t_idx = 1:NumdataPointsOnT
    subplot(2,1,1), hold on
    plot(xData2D,CgData2D(:,t_idx),'*')
    subplot(2,1,2), hold on
    plot(xData2D,CsData2D(:,t_idx),'*')
end

%% Creating dlarray for deeplearning
%Create a grid of tData and xData so that you can the concatenate them to
%create coordinates for each points
[tDataMesh,xDataMesh] = meshgrid(tData2D,xData2D);

%Flatten each array to a row vector
xData = reshape(xDataMesh,[1,NumdataPointsOnT*NumdataPointsOnX]);
tData = reshape(tDataMesh,[1,NumdataPointsOnT*NumdataPointsOnX]);
CgData = reshape(CgData2D,[1,NumdataPointsOnT*NumdataPointsOnX]);
CsData = reshape(CsData2D,[1,NumdataPointsOnT*NumdataPointsOnX]);

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    xData = gpuArray(xData);
    tData = gpuArray(tData);
    CgData = gpuArray(CgData);
    CsData = gpuArray(CsData);
    "GPU"
end

%Creating dlArray to compute
Xdl = dlarray(xData,"CB");
Tdl = dlarray(tData,"CB");
Cgdl = dlarray(CgData,"CB");
Csdl = dlarray(CsData,"CB");

%% Test dlArray
figure('Name', 'Test dlArray')
for t_idx = 1:NumdataPointsOnT
    subplot(2,1,1), hold on
    plot(Xdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),Cgdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
    subplot(2,1,2), hold on
    plot(Xdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),Csdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
end

%% Neural network architecture
numLayers = 9;
numNeurons = 20;

layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(2)]

net = dlnetwork(layers) %convert the network into a dlnetwork object

%% Test net

concat = cat(1,Xdl,Tdl);

[Cg_test1,Cs_test1] = forward(net,concat); %ne marche pas

testData = forward(net,concat);

Cg_test = testData(1,:);
Cs_test = testData(2,:);

%% Training
numEpochs = 200;
solverState = lbfgsState;

lossFcn = @(net) dlfeval(@LossCgCsAnalytical,net,Xdl,Tdl,Cgdl, Csdl);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end

%% testing and ploting with same data as training
XT_test = cat(1, Xdl, Tdl);
CgCs_pred = forward(net, XT_test);
Cg_pred = CgCs_pred(1,:);
Cs_pred = CgCs_pred(2,:);
%%
Cs_pred = Cs_pred * Ke;

figure()
for t_idx=1:NumdataPointsOnT
    t_idx
    subplot(2,1,1), hold on
    plot(xData2D,Cg_pred(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
    subplot(2,1,2), hold on
    plot(xData2D,Cs_pred(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
end
