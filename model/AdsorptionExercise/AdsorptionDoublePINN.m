clc;
clear;

addpath("..\..\losses\AdsorptionLosses\PINN\")

%% Load and prepare data
load TrainingDataPINNWithDataPoints
% List of arrays to convert to dlarray
arraysList = {'xBC1', 'tBC1', 'CgBC1', 'xBC2', 'tBC2', 'CgBC2', ...
              'xIC', 'tIC', 'CgIC', 'CsIC', 'x', 't', 'xd', 'td', 'Cgd', 'Csd'};

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
    fullyConnectedLayer(1)];

% Network for Cg
netCg = dlnetwork(layers) %convert the network into a dlnetwork object
% Network for Cs
netCs = dlnetwork(layers) %convert the network into a dlnetwork object

%% Training
numEpochs = 100;
solverStateCg = lbfgsState;
solverStateCs = lbfgsState;

lossFcnCg = @(net) dlfeval(@LossCg,netCg,netCs,x,t,xIC,tIC,CgIC,xBC1,tBC1,CgBC1,xBC2,tBC2,CgBC2,xd,td,Cgd);
lossFcnCs = @(net) dlfeval(@LossCs,netCg,netCs,x,t,xIC,tIC,CsIC,xd,td,Csd);

monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLossCg","TrainingLossCs"], ...
    Info="Epoch", ...
    XLabel="Epoch");

"start training"
for i = 1:numEpochs
    [netCg, solverStateCg] = lbfgsupdate(netCg,lossFcnCg,solverStateCg, LineSearchMethod="strong-wolfe");
    [netCs, solverStateCs] = lbfgsupdate(netCs,lossFcnCs,solverStateCs, LineSearchMethod="strong-wolfe");

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i, ...
        TrainingLossCg=solverStateCg.Loss, ...
        TrainingLossCs=solverStateCs.Loss);
    if monitor.Stop
        break;
    end
end
