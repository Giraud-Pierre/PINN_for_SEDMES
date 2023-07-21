function [loss,gradients] = LossCsWithCgSol(net,X,T,Cg,XIC,TIC,CsIC,Xd,Td,Csd)
%put Xd, Td, Cgd and Csd equal to empty array ([]) if there is no data

ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;


% Make predictions For the PDEs.
XT = cat(1,X,T);
Cs_pred = forward(net,XT);

% Calculate derivatives of Cg with respect to X and T.
gradientsCg = dlgradient(sum(Cg,"all"),{X,T},EnableHigherDerivatives=true);
Cgx = gradientsCg{1};
Cgt = gradientsCg{2};

% Calculate residuess and enforce PDEs of Cg
residues = Cgt + ug * Cgx + (kg * as) * (Cg - (Cs_pred/Ke)) / epsb;
zeroTarget = zeros(size(residues),"like",residues);
msefCg = l2loss(residues,zeroTarget);


% Calculate derivatives of Cs with respect to T.
Cst = dlgradient(sum(Cs_pred,"all"),T,EnableHigherDerivatives=true);

% Calculate residuess and enforce PDEs of Cs
residues = Cst - (Cg - (Cs_pred/Ke)) * (kg*as) / (1-epsb);
zeroTarget = zeros(size(residues),"like",residues);
msefCs = l2loss(residues,zeroTarget);

% enforce initial conditions
XTIC = cat(1,XIC,TIC);
CsIC_pred = forward(net,XTIC);

mseCsIC = l2loss(CsIC_pred,CsIC);

% enforce data points
if size(Csd) == [0 0] %if there is no datapoints
    mseCsd = 0;
else
    XTd = cat(1,Xd,Td);
    Csd_pred = forward(net,XTd);
    
    mseCsd = l2loss(Csd_pred,Csd);
end

% Calculated loss to be minimized by combining errors.
loss = msefCg + msefCs + mseCsIC + mseCsd;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end