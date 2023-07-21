function [loss,gradients] = LossCs(netCg,netCs,X,T,XIC,TIC,CsIC,Xd,Td,Csd)
%put Xd, Td, Cgd and Csd equal to empty array ([]) if there is no data

ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;


% Make predictions For the PDEs.
XT = cat(1,X,T);
Cg_pred = forward(netCg,XT);
Cs_pred = forward(netCs,XT);

% Calculate derivatives with respect to X and T.
Cst = dlgradient(sum(Cs_pred,"all"),T,EnableHigherDerivatives=true);

% Calculate residuess and enforce PDEs
residues = Cst - (Cg_pred - (Cs_pred/Ke)) * (kg*as) / (1-epsb);
zeroTarget = zeros(size(residues),"like",residues);
msePDE = l2loss(residues,zeroTarget);

% enforce initial conditions
XTIC = cat(1,XIC,TIC);
CsIC_pred = forward(netCs,XTIC);

mseCsIC = l2loss(CsIC_pred,CsIC);

% enforce data points
if size(Xd) == [0 0] %if there is no datapoints
    mseCsd = 0;
else
    XTd = cat(1,Xd,Td);
    Csd_pred = forward(netCs,XTd);
    
    mseCsd = l2loss(Csd_pred,Csd);
end

% Calculated loss to be minimized by combining errors.
lossCs = msePDE + mseCsIC + mseCsd;
loss = lossCs;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,netCs.Learnables);

end