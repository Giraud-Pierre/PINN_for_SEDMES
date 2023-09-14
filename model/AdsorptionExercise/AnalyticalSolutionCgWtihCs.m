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
NumdataPointsOnT = 5;
NumdataPointsOnX = 100;

%Create a grid of tData and xData so that you can the concatenate them to
%create coordinates for each points
X_test = linspace(2*L/10,L,NumdataPointsOnX);
t_test = linspace(2*tend/10,tend,NumdataPointsOnT);
[t_mesh, X_mesh] = meshgrid(t_test,X_test);
%Flatten each array to a row vector
data_X = reshape(X_mesh,[1,numel(X_mesh)]);
data_T = reshape(t_mesh,[1,numel(t_mesh)]);

[data_Cg, data_Cs] = solveAdsorptionPointByPoint(data_X, data_T);
%Compute the analytical solutions of Cg and Cs and return 2 arrays of shape
%NumdataPointsOnX * NumdataPointsOnT which give the concentration function
%of time and space (CgData2D(x,t) is Cg at time tData2D(t) and space xData2D(x))

%% Test SolveAdsorption (analytical solution)
% figure('Name','Test SolveAdsorption solver')
% 
% for t_idx = 1:NumdataPointsOnT
%     subplot(2,1,1), hold on
%     plot(xData2D,CgData2D(:,t_idx),'*')
%     subplot(2,1,2), hold on
%     plot(xData2D,CsData2D(:,t_idx),'*')
% end

%% Creating dlarray for deeplearning
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    Xdl = dlarray(gpuArray(data_X));
    Tdl = dlarray(gpuArray(data_T));
    Cgdl = dlarray(gpuArray(data_Cg));
    Csdl = dlarray(gpuArray(data_Cs));
    "GPU"
else
    Xdl = dlarray(data_X,"CB");
    Tdl = dlarray(data_T,"CB");
    Cgdl = dlarray(data_Cg,"CB");
    Csdl = dlarray(data_Cs,"CB");
end

%% Test dlArray
% figure('Name', 'Test dlArray')
% for t_idx = 1:NumdataPointsOnT
%     subplot(2,1,1), hold on
%     plot(Xdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),Cgdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
%     subplot(2,1,2), hold on
%     plot(Xdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),Csdl(1+NumdataPointsOnX*(t_idx-1):NumdataPointsOnX*t_idx),'*')
% end

%% Neural network architecture
numLayers = 6;
numNeurons = 64;

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

% concat = cat(1,Xdl,Tdl);
% 
% [Cg_test1,Cs_test1] = forward(net,concat); %ne marche pas
% 
% testData = forward(net,concat);
% 
% Cg_test = testData(1,:);
% Cs_test = testData(2,:);

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

CgCs_pred = forward(net, cat(1,Xdl,Tdl));
Cg_pred = CgCs_pred(1,:);
Cs_pred = CgCs_pred(2,:);

errCg = norm(extractdata(Cg_pred) - data_Cg) / norm(data_Cg)
errCs = norm(extractdata(Cs_pred) - data_Cs) / norm(data_Cs)

figure('Name',sprintf('Analytical solution for Cs and Cg in same model, errorCg = %f and errorCs  =%f',errCg, errCs))
subplot(2,2,1)
plot3(data_X,data_T,extractdata(Cg_pred),'*','DisplayName',"Cg predicted")
legend()
subplot(2,2,2)
plot3(data_X,data_T,data_Cg,'*', 'DisplayName', "exact Cg")
legend()
subplot(2,2,3)
plot3(data_X,data_T,extractdata(Cs_pred),'*','DisplayName',"Cs predicted")
legend()
subplot(2,2,4)
plot3(data_X,data_T,data_Cs,'*', 'DisplayName', "exact Cs")
legend()

%% Saving result

% savefig('../../results/AdsorptionExercise/AnalyticalSolutions/AnalyticalSolutionCgCsInOneModel.fig')
% save('../../results/AdsorptionExercise/AnalyticalSolutions/AnalyticalSolutionCgCsInOneModel',"data_X","data_X","data_Cg","Cg_pred","data_Cs","Cs_pred","net")
