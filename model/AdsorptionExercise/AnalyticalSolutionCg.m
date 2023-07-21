clc;
clear;

%% Load and prepare data
load TrainingDataPINNWithDataPoints

% Solve Cg for x and t
Cg = solveAdsorptionPointByPoint(x,t);
%Csd = [];
% List of arrays to convert to dlarray
arraysList = {'xIC', 'tIC', 'CsIC', 'x', 't','Cg', 'xd', 'td', 'Csd'};

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

net = dlnetwork(layers) %convert the network into a dlnetwork object

%% Training
numEpochs = 500;
solverState = lbfgsState;

lossFcn = @(net) dlfeval(@LossCsWithCgSol,net,x,t,Cg,xIC,tIC,CsIC,xd,td,Csd);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

"start training"
for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState, LineSearchMethod="strong-wolfe");

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i, ...
        TrainingLoss=solverState.Loss);
    if monitor.Stop
        break;
    end
end

%% plotting results

xplot = linspace(0,1,100);
tplot = linspace(0,1000,6);
[tplot,xplot] = meshgrid(tplot,xplot);
tplot = reshape(tplot,[1,numel(tplot)]);
xplot = reshape(xplot,[1,numel(xplot)]);
tdlplot = dlarray(tplot,"CB");
xdlplot = dlarray(xplot,"CB");

Csplot = forward(net,cat(1,xdlplot,tdlplot));
[CgExact, CsExact] = solveAdsorptionPointByPoint(xplot,tplot);

err = norm(extractdata(Csplot) - CsExact) / norm(CsExact)

figure()
plot3(xplot,tplot,extractdata(Csplot),'*','DisplayName',"Cs predicted")
hold on
plot3(xplot,tplot,CsExact,'*', 'DisplayName', "exact Cs")
legend()