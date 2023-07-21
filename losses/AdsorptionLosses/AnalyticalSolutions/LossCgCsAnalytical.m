function [loss,gradients] = LossCgCsAnalytical(net,X,T,Cg,Cs)
ke = 10;
% Make predictions For the PDEs.
XT = cat(1,X,T);
CgCs_pred = forward(net,XT);
Cg_pred = CgCs_pred(1,:);
Cs_pred = CgCs_pred(2,:);
    
mseCg = l2loss(Cg_pred,Cg);
mseCs = l2loss(Cs_pred,Cs);


% Calculated loss to be minimized by combining errors.
loss = mseCg + mseCs;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end