clear all;
close all;

C0 = 0;
Cin = 1;
tau = 1;
Ke = 1;

output = sim('ODE_reaction_Simulink.slx');
C = output.simout.data;
t = output.simout.Time;

C_sol = (Cin/(1 + Ke * tau))*(1 - exp(-t.*(Ke + (1/tau))));

plot(t,C)
hold on
plot(t,C_sol)
hold off

error = mean((C_sol - C).^2)


save("ODE_reaction_data.mat","C","C_sol","t","error")