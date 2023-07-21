function [loss,gradients] = LossCgWithCsSol(net,X,T,Cs,XIC,TIC,CgIC,Xd,Td,Cgd)
%put Xd, Td, Cgd and Csd equal to empty array ([]) if there is no data

ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;


% Make predictions For the PDEs.
XT = cat(1,X,T);
Cg_pred = forward(net,XT);

% Calculate derivatives of Cg with respect to X and T.
gradientsCg = dlgradient(sum(Cg_pred,"all"),{X,T},EnableHigherDerivatives=true);
Cgx = gradientsCg{1};
Cgt = gradientsCg{2};

% Calculate residuess and enforce PDEs of Cg
residues = Cgt + ug * Cgx + (kg * as) * (Cg_pred - (Cs/Ke)) / epsb;
zeroTarget = zeros(size(residues),"like",residues);
msefCg = l2loss(residues,zeroTarget);


% Calculate derivatives of Cs with respect to T.
Cst = dlgradient(sum(Cs,"all"),T,EnableHigherDerivatives=true);

% Calculate residuess and enforce PDEs of Cs
residues = Cst - (Cg_pred - (Cs/Ke)) * (kg*as) / (1-epsb);
zeroTarget = zeros(size(residues),"like",residues);
msefCs = l2loss(residues,zeroTarget);

% enforce initial conditions
XTIC = cat(1,XIC,TIC);
CgIC_pred = forward(net,XTIC);

mseCgIC = l2loss(CgIC_pred,CgIC);

% enforce data points
if size(Cgd) == [0 0] %if there is no datapoints
    mseCgd = 0;
else
    XTd = cat(1,Xd,Td);
    Cgd_pred = forward(net,XTd);
    
    mseCgd = l2loss(Cgd_pred,Cgd);
end

% Calculated loss to be minimized by combining errors.
loss = msefCg + msefCs + mseCgIC + mseCgd;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end