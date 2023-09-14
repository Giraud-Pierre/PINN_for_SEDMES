function [loss,gradients] = LossCgCs(net,X,T,XIC,TIC,CgIC,CsIC,X0,T0,Cg0,Xm,Tm,Cgm,Xd,Td,Cgd,Csd)
%put Xd, Td, Cgd and Csd equal to empty array ([]) if there is no data

ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;


% Make predictions For the PDEs.
XT = cat(1,X,T);
CgCs_pred = forward(net,XT);
Cg_pred = CgCs_pred(1,:);
Cs_pred = CgCs_pred(2,:);

% Calculate derivatives with respect to X and T.
gradientsCg = dlgradient(sum(Cg_pred,"all"),{X,T},EnableHigherDerivatives=true);
Cgx = gradientsCg{1};
Cgt = gradientsCg{2};
gradientsCs = dlgradient(sum(Cs_pred,"all"),{X,T},EnableHigherDerivatives=true);
Cst = gradientsCs{2};

% Calculate mseF. Enforce PDEs
fCg = Cgt + ug * Cgx + (kg * as) * (Cg_pred - (Cs_pred/Ke)) / epsb;
zeroTarget = zeros(size(fCg),"like",fCg);
mseFCg = l2loss(fCg,zeroTarget) / mean(Cgd);

fCs = Cst - (Cg_pred - (Cs_pred/Ke)) * (kg*as) / (1-epsb);
zeroTarget = zeros(size(fCs),"like",fCs);
mseFCs = l2loss(fCs,zeroTarget) / mean(Csd);

% enforce initial conditions
XTIC = cat(1,XIC,TIC);
CgCsIC_pred = forward(net,XTIC);
CgIC_pred = CgCsIC_pred(1,:);
CsIC_pred = CgCsIC_pred(2,:);

mseCgIC = l2loss(CgIC_pred,CgIC);
mseCsIC = l2loss(CsIC_pred,CsIC);

% enforce boundary condition 1
XT0 = cat(1,X0,T0);
CgCs0_pred = forward(net,XT0);
Cg0_pred = CgCs0_pred(1,:);

mseCg0 = l2loss(Cg0_pred,Cg0) / mean(Cgd);

% enforce boundary condition 2
XTm = cat(1,Xm,Tm);
CgCsm_pred = forward(net,XTm);
Cgm_pred = CgCsm_pred(1,:);

Cgm_x_pred = dlgradient(sum(Cgm_pred,"all"),Xm,EnableHigherDerivatives=true);

mseCgm = l2loss(Cgm_x_pred,Cgm) / mean(Cgd);

% enforce data points
if size(Xd) == [0 0] %if there is no datapoints
    mseCgd = 0;
    mseCsd = 0;
else
    XTd = cat(1,Xd,Td);
    CgCsd_pred = forward(net,XTd);
    Cgd_pred = CgCsd_pred(1,:);
    Csd_pred = CgCsd_pred(2,:);
    
    mseCgd = l2loss(Cgd_pred,Cgd) / mean(Cgd);
    mseCsd = l2loss(Csd_pred,Csd) / mean(Csd);
end

% Calculated loss to be minimized by combining errors.
loss = mseFCg + mseFCs + mseCgIC + mseCsIC + mseCg0 + mseCgm + 6* (mseCgd + mseCsd);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end