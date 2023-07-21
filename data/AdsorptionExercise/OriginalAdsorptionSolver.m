clear; 
clc;
format long e

L = 1.0;
ug = 0.01;
epsb = 0.5;
kg = 0.0001;
Ke = 10;
dp = 0.005;
as = 6*(1-epsb)/dp;
tend = 1000;

N = 100;
M = 5;
x = linspace(0, L, N+1);
t = linspace(0, tend, M+1);

figure, hold on
title('Analytical solution:')
xlabel('x/L [-]');
h = legend('show','location','best');
set(h,'FontSize',12);
Cg_all = zeros(M,N);
Cs_all = zeros(M,N);
integral_all = zeros(M,N);
g_all = zeros(M,N);
tau_all = zeros(M,N);
xi_all = zeros(M,N);

tic
for j=1:M+1
    for i=1:N+1
        tau = kg*as/((1-epsb)*Ke)*(t(j) - x(i)/ug);
        xi = kg*as/(epsb*ug)*x(i);
        fnc = @(u) exp(-u).*besseli(0, sqrt(4*tau*u), 1)./exp(-abs(real(sqrt(4*tau*u))));
        I = integral(fnc, 0 , xi);
        e = exp(-tau);
        g = besseli(0, sqrt(4*tau*xi))*exp(-xi);
        Cg(i) = real(1 - e*I);
        Cs(i) = real(Ke*(1 - e*(I + g)));
        Cg_all(j,i) = Cg(i);
        Cs_all(j,i) = Cs(i);
        integral_all(j,i) = real(I);
        g_all(j,i) = real(g);
        tau_all(j,i) = tau;
        xi_all(j,i) = xi;
    end
    Cg(isinf(Cg)) = 0;
    Cs(isinf(Cs)) = 0;

    subplot(2,1,1), hold on
    yyaxis left, hold on
    plot(x/L,Cg,'-x','LineWidth',1,'MarkerSize',1,'DisplayName',num2str(t(j))) 
    ylabel('Cg/Cgin [-]');
    ylim([0 1]);

    yyaxis right, hold on
    subplot(2,1,2), hold on
    plot(x/L,Cs,'-x','LineWidth',1,'MarkerSize',1,'DisplayName',num2str(t(j))) 
    ylabel('Cs/Cgin [-]');    
    ylim([0 Ke]);
end
toc
save("adsorptionData.mat","Cs_all","Cg_all","t","x", "integral_all", "g_all", "tau_all", "xi_all")
