clear; 
clc;
format long e

addpath("..\..\losses\SimpleCosSinTest\")
addpath("..\..\data\SimpleCosSinTest\")

load CosSinData

%% Creating dlarray for deeplearning
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    Xdl = gpuArray(dlarray(data_X,"CB"));
    Tdl = gpuArray(dlarray(data_T,"CB"));
    Udl = gpuArray(dlarray(U,"CB"));
    "GPU"
else
    Xdl = dlarray(data_X,"CB");
    Tdl = dlarray(data_T,"CB");
    Udl = dlarray(U,"CB");
end

%% Neural network architecture
numLayers = 9;
numNeurons = 32;

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

net = dlnetwork(layers) %convert the network into a dlnetwork object

%% Training with L BFGS
% numEpochs = 1000;
% solverState = lbfgsState;
% 
% lossFcn = @(net) dlfeval(@LossU,net, Xdl, Tdl, Udl);
% 
% monitor = trainingProgressMonitor( ...
%     Metrics="TrainingLoss", ...
%     Info="Epoch", ...
%     XLabel="Epoch");
% 
% for i = 1:numEpochs
%     [net, solverState] = lbfgsupdate(net,lossFcn,solverState, LineSearchMethod="weak-wolfe");
% 
%     updateInfo(monitor,Epoch=i);
%     recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
% end

%% Training with adam
numEpochs = 2000;
initial_learning_rate = 0.01;
decay_rate = 0.001;
if exist('averageGrad','var') == false
    averageGrad = [];
end
if exist('averageSqGrad', 'var') == false
    averageSqGrad = [];
end

lossFcn = @(net) dlfeval(@LossU,net, Xdl, Tdl, Udl);
lossLog = zeros(numEpochs);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

learning_rate = initial_learning_rate;
for i = 1:numEpochs
    [loss, gradients] = lossFcn(net);
    lossLog(i) = loss;

    learning_rate = initial_learning_rate / (1 + decay_rate * i + 10);

    [net, averageGrad, averageSqGrad] = adamupdate(net,gradients,averageGrad, averageSqGrad, i, learning_rate);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=loss);
end

%% testing and ploting with same data as training
X_test = linspace(0,2 * pi,100);
t_test = linspace(0,2 * pi,6);
[X, T] = meshgrid(X_test,t_test);
%Flatten each array to a row vector
X = reshape(X,[1,numel(X)]);
T = reshape(T,[1,numel(T)]);
Xdl = dlarray(X,"CB");
Tdl = dlarray(T,"CB");

% compute error
U_pred = forward(net,cat(1,Xdl, Tdl));
U_exact = solveU(X,T);

err = norm(extractdata(U_pred) - U_exact) / norm(U_exact)

%plot the result
figure('Name',sprintf("AnalyticalSolution sin*cos, error = %f",err))
plot3(X,T,extractdata(U_pred),'*','DisplayName',"U predicted")
hold on
plot3(X,T,U_exact,'*', 'DisplayName', "exact U")
legend()

%% Saving result

savefig('../../results/CosSin/Sin3xCos2t_PINNlike_withAdam.fig')
save('../../results/CosSin/Sin3xCos2t_PINNlike_withAdam.mat',"X","T","U_exact","U_pred","net")
