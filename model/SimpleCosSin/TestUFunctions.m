clear; 
clc;
format long e

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

%% Test dlArray
figure()

%plot random
% subplot(2,1,1)
% scatter(extractdata(Xdl), extractdata(Udl), 10, extractdata(Tdl))
% subplot(2,1,2)
% scatter(extractdata(Tdl), extractdata(Udl), 10, extractdata(Xdl))

% Plot linspace
for t_index=1:numel(data_T_linspace)
    hold on
    start = 1 + (t_index-1)*numel(data_X_linspace);
    stop = t_index*numel(data_X_linspace);
    plot(Xdl(start:stop),Udl(start:stop))
end


%% Neural network architecture
numLayers = 10;
numNeurons = 10;

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

%% Test net

concat = cat(1,Xdl,Tdl);

U = forward(net,concat);

%% Training with L BFGS
numEpochs = 1000;
solverState = lbfgsState;

lossFcn = @(net) dlfeval(@LossU,net, Xdl, Tdl, Udl);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState, LineSearchMethod="weak-wolfe");

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end

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

figure()
for t_idx=1:numel(t_test)
    T = ones(1,numel(X_test)) * t_test(t_idx);
    XT = cat(1,X_test, T);
    U_pred = forward(net,dlarray(XT,"CB"));
    U_exact = solveU(X_test,T);

    subplot(2,1,1), hold on
    plot(X_test,U_pred)
    subplot(2,1,2), hold on
    plot(X_test,U_exact)
end
