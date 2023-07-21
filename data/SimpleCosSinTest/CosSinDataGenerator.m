clear; 
clc;
format long e

%% Get data for initial condition and collocation points
%Computing Data Point

% numInternalCollocationPoints = 200;
% 
% points = rand(2,numInternalCollocationPoints);
% 
% data_X = points(1,:) * 2 * pi;
% data_T = points(2,:) * 2 * pi;

data_X_linspace = linspace(0,2 * pi,100);
data_T_linspace = linspace(0,2 * pi,10);
[Data_T,Data_X] = meshgrid(data_T_linspace,data_X_linspace);

data_X = reshape(Data_X,[1,numel(Data_X)]);
data_T = reshape(Data_T,[1,numel(Data_T)]);

U = solveU(data_X, data_T);


%% Test SolveAdsorption (analytical solution)
figure()
% Plot random
% subplot(2,1,1)
% scatter(data_X, U,10,data_T)
% subplot(2,1,2)
% scatter(data_T, U,10,data_X)

% Plot linspace
for t_index=1:numel(data_T_linspace)
    hold on
    start = 1 + (t_index-1)*numel(data_X_linspace);
    stop = t_index*numel(data_X_linspace);
    plot(data_X(start:stop),U(start:stop))
end

%% Save Workspace

save("CosSinData.mat", "data_X","data_T","U")