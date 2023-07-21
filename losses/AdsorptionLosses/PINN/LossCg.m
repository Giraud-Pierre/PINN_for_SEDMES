function [loss,gradients] = LossCg(netCg,netCs,X,T,XIC,TIC,CgIC,X0,T0,Cg0,Xm,Tm,Cgm,Xd,Td,Cgd)
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
gradientsCg = dlgradient(sum(Cg_pred,"all"),{X,T},EnableHigherDerivatives=true);
Cgx = gradientsCg{1};
Cgt = gradientsCg{2};

% Calculate residuess and enforce PDEs
residues = Cgt + ug * Cgx + (kg * as) * (Cg_pred - (Cs_pred/Ke)) / epsb;
zeroTarget = zeros(size(residues),"like",residues);
msePDE = l2loss(residues,zeroTarget);

% enforce initial conditions
XTIC = cat(1,XIC,TIC);
CgIC_pred = forward(netCg,XTIC);

mseCgIC = l2loss(CgIC_pred,CgIC);

% enforce boundary condition 1
XT0 = cat(1,X0,T0);
Cg0_pred = forward(netCg,XT0);

mseCg0 = l2loss(Cg0_pred,Cg0);

% enforce boundary condition 2
XTm = cat(1,Xm,Tm);
Cgm_pred = forward(netCg,XTm);

Cgm_x_pred = dlgradient(sum(Cgm_pred,"all"),Xm,EnableHigherDerivatives=true);

mseCgm = l2loss(Cgm_x_pred,Cgm);

% enforce data points
if size(Xd) == [0 0] %if there is no datapoints
    mseCgd = 0;
else
    XTd = cat(1,Xd,Td);
    Cgd_pred = forward(netCg,XTd);
    
    mseCgd = l2loss(Cgd_pred,Cgd);
end

% Calculated loss to be minimized by combining errors.
lossCg = msePDE + mseCgIC + mseCg0 + mseCgm + mseCgd;
loss = lossCg;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,netCg.Learnables);

end