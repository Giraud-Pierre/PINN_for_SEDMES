function transient_plug_flow
clear all; clc; close all

%% Definition of the function
    function dCdt = reaction_lines(t,C)
    % we use vectorized operations to define the odes at each node
    % point.
    dCdt = [0; % C1 does not  change with time, it is the entrance of the pfr
           -vo*diff(C)./diff(V)-k*C(2:end).^2];
    end

%% Initialization
Ca0 = 2; % Entering concentration
vo = 2; % volumetric flow rate
volume = 20;  % total volume of reactor, spacetime = 10
k = 1;  % reaction rate constant

N = 100; % number of points to discretize the reactor volume on

init = zeros(N,1); % Concentration in reactor at t = 0
init(1) = Ca0; % concentration at entrance

V = linspace(0,volume,N)'; % discretized volume elements, in column form
tspan = [0 20];

%% Execution
[t,C] = ode15s(@reaction_lines,tspan,init);

%% PLoting
plot(t,C(:,end))
xlabel('time')
ylabel('Concentration at exit')

%% Animation
filename = 'transient-pfr.gif'; % file to store image in
for i=1:5:length(t) % only look at every 5th time step
    plot(V,C(i,:));
    ylim([0 2])
    xlabel('Reactor volume')
    ylabel('C_A')
    text(0.1,1.9,sprintf('t = %1.2f',t(i))); % add text annotation of time step
    frame = getframe(1);
    im = frame2im(frame); %convert frame to an image
    [imind,cm] = rgb2ind(im,256);

    % now we write out the image to the animated gif.
    if i == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end
close all; % makes sure the plot from the loop above doesn't get published

%% add steady state
    function dCdV = pfr(V,C)
        dCdV = 1/vo*(-k*C^2);
    end

figure; hold all
vspan = [0 20];
[V_ss Ca_ss] = ode45(@pfr,vspan,2);
plot(V_ss,Ca_ss,'k--','linewidth',2)
plot(V,C(end,:),'r') % last solution of transient part
legend 'Steady state solution' 'transient solution at t=100'

end