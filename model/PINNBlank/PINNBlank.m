clear; 
clc;
format long e

addpath("..\..\data\PINNBlankData\")
addpath("..\..\losses\AdsorptionLosses\AnalyticalSolutions\")

%% Data
% load BurgerData
% load SinCos
load AdsorptionDataCg
% load AdsorptionDataCs

% Contient un tableau x, un tableau t et un tableau y
% Ces 3 tableaux doivent possèder une forme (1, N), N étant le nombre de
% points du dataset

%% Network
numLayers = 6;
numNeurons = 64;
% numNeurons = 128;

layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [layers  fullyConnectedLayer(numNeurons)  tanhLayer];
end

layers = [layers  fullyConnectedLayer(1)]

net = dlnetwork(layers)

%% Initialize dlArrays
X = dlarray(x,"CB");
T = dlarray(t,"CB");
Y = dlarray(y,"CB");

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    X = gpuArray(X);
    T = gpuArray(T);
    Y = gpuArray(Y);
end

%% Training
solverState = lbfgsState;
numEpochs = 500;

lossFcn = @(net) dlfeval(@LossAnalytical,net,X,T,Y);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState, LineSearchMethod='weak-wolfe');

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end

%% Testing and ploting
XT = cat(1,X,T);
y_predict = extractdata(forward(net,XT));

err = norm(y_predict - y) / norm(y)

figure()
subplot(2,1,1)
plot3(x,t,y_predict,'*', "DisplayName", "Cg predicted")
hold on
plot3(x,t,y,'x', "DisplayName", "Exact Cg")
legend()
grid on
