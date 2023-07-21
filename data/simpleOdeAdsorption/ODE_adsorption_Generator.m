clear all;
close all;

k=1;

Cin = 1;
tau = 100;
Ke = 10;
epsb = 0.5;
dp = 0.005;
as = 6*(1-epsb)/dp;
kg = 0.0001;

output = sim('ODE_adsorption_Simulink.slx');
Cg = output.Cg.data;
Cs = output.Cs.data;
t = output.tout;

plot(output.Cg)
hold on
plot(output.Cs)
hold off

save("ODE_adsorption_data.mat","Cg","Cs","t")