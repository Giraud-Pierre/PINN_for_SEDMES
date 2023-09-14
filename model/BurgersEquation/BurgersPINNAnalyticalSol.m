clear; 
clc;
format long e

addpath("..\..\losses\BurgersEquationLosses\")
addpath("..\..\data\BurgersEquation\")

nu = 0.01/pi;

%% Data
load BurgersEquationTrainingData

arraysList = {'dataXAnaSol', 'dataTAnaSol', 'DataUAnaSol', 'X0', 'T0', 'U0'};

% Loop through each array and convert it to dlarray or gpu array if gpu is
% enabled
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    for i = 1:numel(arraysList)
        arrayName = arraysList{i};
        eval(arrayName + " = gpuArray(dlarray(" + arrayName + ',"CB"));');
    end
else
    for i = 1:numel(arraysList)
        arrayName = arraysList{i};
        eval(arrayName + " = dlarray(" + arrayName + ',"CB");');
    end
end

%% Network
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
    fullyConnectedLayer(1)]

net = dlnetwork(layers)

%% Training
solverState = lbfgsState;
numEpochs = 500;

lossFcn = @(net) dlfeval(@BurgerAnalyticalLoss,net,dataXAnaSol,dataTAnaSol,DataUAnaSol,X0,T0,U0);

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
%Create a grid of x and t to test the model
tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);
[X, T] = meshgrid(XTest,tTest);
%Flatten each array to a row vector
X = reshape(X,[1,numel(X)]);
T = reshape(T,[1,numel(T)]);
Xdl = dlarray(X,"CB");
Tdl = dlarray(T,"CB");

%Compute the errors
Upredict = forward(net,cat(1,Xdl,Tdl));
for index=1:numel(X)
    Uexact(1,index) = solveBurgers(X(index),T(index),nu);
end

err = norm(extractdata(Upredict) - Uexact) / norm(Uexact)

%plot the result
figure('Name',sprintf("Model with analytical solution for Burger's equation, error = %f",err))
plot3(X,T,extractdata(Upredict),'*','DisplayName',"U predicted")
hold on
plot3(X,T,Uexact,'*', 'DisplayName', "exact U")
legend()


%% Saving result

%savefig('../../results/BurgersEquation/BurgersAnalytic.fig')
%save('../../results/BurgersEquation/BurgersAnalytic.mat',"X","T","Uexact","Upredict","net")
