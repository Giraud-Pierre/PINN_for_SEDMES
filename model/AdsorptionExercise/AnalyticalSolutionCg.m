clc;
clear;

addpath("..\..\data\AdsorptionExercise\")
addpath("..\..\losses\AdsorptionLosses\AnalyticalSolutions\")

%% Load and prepare data

x = linspace(0,1,300);
t = linspace(100,1000,10);
[x,t] = meshgrid(x,t);
t = reshape(t,[1,numel(t)]);
x = reshape(x,[1,numel(x)]);

% Solve Cg for x and t
[Cg,Cs] = solveAdsorptionPointByPoint(x,t);
% List of arrays to convert to dlarray
arraysList = {'x', 't','Cg'};

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
numNeurons = 128;

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
numEpochs = 100;
solverState = lbfgsState;

lossFcn = @(net) dlfeval(@LossAnalytical,net,x,t,Cg);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

"start training"
for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState)%, LineSearchMethod="strong-wolfe");

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i, ...
        TrainingLoss=solverState.Loss);
    if monitor.Stop
        break;
    end
end

%% plotting results

xplot = linspace(0,1,100);
tplot = linspace(100,1000,5);
[tplot,xplot] = meshgrid(tplot,xplot);
tplot = reshape(tplot,[1,numel(tplot)]);
xplot = reshape(xplot,[1,numel(xplot)]);
tdlplot = dlarray(tplot,"CB");
xdlplot = dlarray(xplot,"CB");

Cgpredict = forward(net,cat(1,xdlplot,tdlplot));
[CgExact, CsExact] = solveAdsorptionPointByPoint(xplot,tplot);

err = norm(extractdata(Cgpredict) - CgExact)/ norm(CgExact)

figure('Name', sprintf('Analytical solution Cg, error = %f',err))
plot3(xplot,tplot,extractdata(Cgpredict),'*','DisplayName',"Cg predicted")
hold on
plot3(xplot,tplot,CgExact,'*', 'DisplayName', "exact Cg")
legend()

%% Saving result

%savefig('../../results/AdsorptionExercise/AnalyticalSolutions/Analytical_solution_Cg.fig')
%save('../../results/AdsorptionExercise/AnalyticalSolutions/Analytical_solution_Cg.mat',"xplot","tplot","CgExact","Cgpredict","net")