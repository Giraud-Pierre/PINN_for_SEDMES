function [loss,gradients] = BurgerAnalyticalLoss(net,X,T,U,X0,T0,U0)

% Enforce collocation points.
XT = cat(1,X,T);
U_pred = forward(net,XT);

mseF = l2loss(U_pred,U);

% Calculate mseU. Enforce initial and boundary conditions.
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
loss = mseF + mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end