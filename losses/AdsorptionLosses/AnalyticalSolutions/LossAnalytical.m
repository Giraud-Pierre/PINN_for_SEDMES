function [loss,gradients] = LossAnalytical(net,X,T,Y)
XT = cat(1,X,T);
C_pred = forward(net,XT);
mseC = l2loss(C_pred,Y) / mean(Y);

% Calculated loss to be minimized by combining errors.
loss = mseC;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end