function [loss,gradients] = LossU(net,X,T,U)
% enforce the colocation points
XT = cat(1,X,T);
U_pred = forward(net,XT);

loss = l2loss(U_pred,U);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end