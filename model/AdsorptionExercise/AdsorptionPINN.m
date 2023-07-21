clear; 
clc;
format long e

addpath("..\..\losses\AdsorptionLosses\PINN\")
addpath("..\..\data\AdsorptionExercise\")

load TrainingDataPINNWithDataPoints
%% Changing data to dlarray
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

%% PINN with no data points
% %% Train the network 
% numEpochs = 0;
% solverState = lbfgsState;
% xd_nodata = gpuArray(dlarray([])) 
% 
% lossFcn = @(net) dlfeval(@LossCgCs,net,x,t,xIC,tIC,CgIC,CsIC, ...
%     xBC1,tBC1,CgBC1,xBC2,tBC2,CgBC2,xd_nodata,td,Cgd,Csd);
% 
% 
% monitor = trainingProgressMonitor( ...
%     Metrics="TrainingLoss", ...
%     Info="Epoch", ...
%     XLabel="Epoch");
% 
% for i = 1:numEpochs
%     [net, solverState] = lbfgsupdate(net,lossFcn,solverState);
% 
%     updateInfo(monitor,Epoch=i);
%     recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
% end
% 
% %% Testing the PINN and ploting the results
% % Space and Time axis
% numPredictions = 100;
% numTime = 5;
% xplot = linspace(0, L, numPredictions);
% tplot = linspace(0, tend, numTime+1);
% 
% % Calculate target.
% [Cg_exact, Cs_exact] = solveAdsorption(xplot,tplot);
% 
% XTest = dlarray(xplot,"CB");
% 
% figure('name', 'no data points')
% tiledlayout("flow")
% 
% for i=1:length(tplot)
%     T = tplot(i);
%     TTest = T*ones(1,numPredictions);
%     TTest = dlarray(TTest,"CB");
% 
%     % Make predictions.
%     XTTest = cat(1,XTest,TTest);
%     CgCs_Pred = forward(net,XTTest);
%     Cg_pred = CgCs_Pred(1,:);
%     Cs_pred = CgCs_Pred (2,:);
% 
%     % Calculate error.
%     Cg_pred = extractdata(Cg_pred);
%     Cs_pred = extractdata(Cs_pred);
%     errCg = norm(Cg_pred - Cg_exact(:,i)) / norm(Cg_exact(:,i));
% 
%     % Plot prediction.
%     nexttile
%     plot(XTest,Cg_pred,"-",LineWidth=2);
%     ylim([-0.1, 1.1])
% 
%     % Plot target.
%     hold on
%     plot(XTest, Cg_exact(:,i),"--",LineWidth=2)
%     hold off
% 
%     title("Cg: t = " + T + ", Error = " + gather(errCg));
% 
%     errCs = norm(Cs_pred - Cs_exact(:,i)) / norm(Cs_exact(:,i));
% 
%         % Plot prediction.
%     nexttile
%     plot(XTest,Cs_pred,"-",LineWidth=2);
%     ylim([-1, 11])
% 
%     % Plot target.
%     hold on
%     plot(XTest, Cs_exact(:,i),"--",LineWidth=2)
%     hold off
% 
%     title("Cs: t = " + T + ", Error = " + gather(errCs));
% end
% 
% legend(["Prediction" "Target"])

%% PINN with data points
%% Train the network 
numEpochs = 300;
solverState = lbfgsState;

lossFcn = @(net) dlfeval(@LossCgCs,net,x,t,xIC,tIC,CgIC,CsIC, ...
     xBC1,tBC1,CgBC1,xBC2,tBC2,CgBC2,xd,td,Cgd,Csd);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end

%% Testing the PINN and ploting the results
% Space and Time axis
L = 1;
tend = 1000;
numPredictions = 100;
numTime = 5;
x = linspace(0, L, numPredictions);
t = linspace(0, tend, numTime+1);

% Calculate target.
[Cg_exact, Cs_exact] = solveAdsorption(x,t);

XTest = dlarray(x,"CB");

figure('name', '24 data points')
tiledlayout("flow")

for i=1:length(t)
    T = t(i);
    TTest = T*ones(1,numPredictions);
    TTest = dlarray(TTest,"CB");

    % Make predictions.
    XTTest = cat(1,XTest,TTest);
    CgCs_Pred = forward(net,XTTest);
    Cg_pred = CgCs_Pred(1,:);
    Cs_pred = CgCs_Pred (2,:);

    % Calculate error.
    Cg_pred = extractdata(Cg_pred);
    Cs_pred = extractdata(Cs_pred);
    errCg = norm(Cg_pred - Cg_exact(:,i))' / norm(Cg_exact(:,i))';

    % Plot prediction.
    nexttile
    plot(XTest,Cg_pred,"-",LineWidth=2);
    ylim([-0.1, 1.1])

    % Plot target.
    hold on
    plot(XTest, Cg_exact(:,i)',"--",LineWidth=2)
    hold off

    title("Cg: t = " + T + ", Error = " + gather(errCg));

    errCs = norm(Cs_pred - Cs_exact(:,i)') / norm(Cs_exact(:,i)');

        % Plot prediction.
    nexttile
    plot(XTest,Cs_pred,"-",LineWidth=2);
    ylim([-1, 11])

    % Plot target.
    hold on
    plot(XTest, Cs_exact(:,i)',"--",LineWidth=2)
    hold off

    title("Cs: t = " + T + ", Error = " + gather(errCs));
end

legend(["Prediction" "Target"])
