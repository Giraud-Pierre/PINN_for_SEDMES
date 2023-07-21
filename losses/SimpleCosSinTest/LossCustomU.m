function [loss,gradients] = LossCustomU(net,X,T,U)
% enforce the colocation points
XT = cat(1,X,T);
U_pred = forward(net,XT);

loss = mean((U_pred - U).^2) / mean(U);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end