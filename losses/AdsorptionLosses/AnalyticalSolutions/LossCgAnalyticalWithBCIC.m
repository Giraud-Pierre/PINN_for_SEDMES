function [loss,gradients] = LossCgAnalyticalWithBCIC(net,XIC,TIC,CgIC,XBC1,TBC1,CgBC1,X,T,Cg)
% Enforce initial condition
XTIC = cat(1,XIC,TIC);
CgIC_pred = forward(net,XTIC);

mseIC = l2loss(CgIC_pred,CgIC);

% enforce boundary condition 1
XTBC1 = cat(1,XBC1,TBC1);
CgBC1_pred = forward(net,XTBC1);

mseBC1 = l2loss(CgBC1_pred,CgBC1);

% enforce the colocation points
XTCg = cat(1,X,T);
Cg_pred = forward(net,XTCg);

mseCg = l2loss(Cg_pred,Cg);

% Calculated loss to be minimized by combining errors.
loss = mseCg + mseIC + mseBC1;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end