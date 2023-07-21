function [loss,gradients] = LossCsAnalyticalWithBCIC(net,XIC,TIC,CsIC,X,T,Cs)
% Enforce initial condition
XTIC = cat(1,XIC,TIC);
CsIC_pred = forward(net,XTIC);

mseIC = l2loss(CsIC_pred,CsIC);

% enforce the colocation points
XTCs = cat(1,X,T);
Cs_pred = forward(net,XTCs);
    
mseCg = l2loss(Cs_pred,Cs);

% Calculated loss to be minimized by combining errors.
loss = mseCg + mseIC;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end