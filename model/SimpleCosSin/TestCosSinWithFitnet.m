% Function U with fitnet
clear; 
clc;
format long e

%% Generate training data
x = linspace(0, 2*pi, 100);
t = linspace(0, 2*pi, 10);
[X, T] = meshgrid(x, t);
U = sin(3*X) .* cos(2*T);
%U = sin((3*X) .* (2*T));

%% Create a feedforward neural network
net = fitnet([16, 16, 16]);  % Specify the number of neurons in each hidden layer
net.trainParam.max_fail = 10;
%% Train the neural network
net = train(net, [X(:), T(:)]', U(:)');

%% Generate test data
x_test = linspace(0, 2*pi, 100);
t_test = linspace(0, 2*pi, 6);
[X_test, T_test] = meshgrid(x_test, t_test);

%% Evaluate the trained network on test data
U_pred = net([X_test(:), T_test(:)]');

% Reshape the predicted output to match the size of the test data grid
U_pred_reshape = reshape(U_pred, size(X_test));

%% Compute the mean square error (MSE)
mse = mean(mean((U_pred_reshape - sin(3*X_test) .* cos(2*T_test)).^2))
% mse = mean(mean((U_pred_reshape - sin(3*X_test .*(2*T_test))).^2))

%% Plot the original and predicted outputs
figure('Name',sprintf("sin3x * cos(2t) using fitnet, error = %f",mse));
subplot(2, 1, 1);
surf(x_test, t_test, U_pred_reshape);
title('Predicted Output');
xlabel('x');
ylabel('t');
zlabel('u');

U_exact = sin(3*X_test) .* cos(2*T_test);
subplot(2, 1, 2);
surf(x_test, t_test, U_exact);
% surf(x_test, t_test, sin(3*X_test .* (2*T_test)));
title('Original Output');
xlabel('x');
ylabel('t');
zlabel('u');

%% Saving result

savefig('../../results/CosSin/Sin3xCos2t_withFitnet.fig')
save('../../results/CosSin/Sin3xCos2t_PINNlike_withFitnet.mat',"X","T","U_exact","U_pred","net")

