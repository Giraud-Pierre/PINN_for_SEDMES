function [loss,gradients] = LossAnalytical(net,X,T,Y)
XT = cat(1,X,T);
Cg_pred = forward(net,XT);
mseCg = l2loss(Cg_pred,Y);

% Calculated loss to be minimized by combining errors.
loss = mseCg;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end