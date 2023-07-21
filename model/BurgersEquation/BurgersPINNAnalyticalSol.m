clear; 
clc;
format long e

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
%%
X = dlarray(dataX,"BC");
T = dlarray(dataT,"BC");
U = dlarray(DataU,"CB");
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0,"CB");

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    X = gpuArray(X);
    T = gpuArray(T);
    U = gpuArray(U);
    X0 = gpuArray(X0);
    T0 = gpuArray(T0);
    U0 = gpuArray(U0);
    "GPU"
end

%% Training
solverState = lbfgsState;
numEpochs = 500;

lossFcn = @(net) dlfeval(@modelLoss,net,X,T,U,X0,T0,U0);

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
tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);
XTest = dlarray(XTest,"CB");

figure
tiledlayout("flow")

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);
    TTest = dlarray(TTest,"CB");

    % Make predictions.
    XTTest = cat(1,XTest,TTest);
    UPred = forward(net,XTTest);

    % Calculate target.
    UTest = solveBurgers(extractdata(XTest),t,0.01/pi);

    % Calculate error.
    UPred = extractdata(UPred);
    err = norm(UPred - UTest) / norm(UTest);

    % Plot prediction.
    nexttile
    plot(XTest,UPred,"-",LineWidth=2);
    ylim([-1.1, 1.1])

    % Plot target.
    hold on
    plot(XTest, UTest,"--",LineWidth=2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

legend(["Prediction" "Target"])

