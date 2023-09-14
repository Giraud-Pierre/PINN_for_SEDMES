# PINN on matlab: Model

This folder contain the actual building, training and testing of the PINNs regarding the adsorption exercise.

At first, it was tried to fit the analytical solution of this exercise using a neural network with a PINN like structure:
- AnalyticalSolutionCg tries to fit Cg
- AnalyticalSolutionCs tries to fit Cs
- AnalyticalSolutionCgCs tries to fit Cg and Cs together
- AnalyticalSolutionCgWithBCIC tries to fit Cg with additionnal data on the boundary condition and the initial condition like a normal PINN would.

Then an hybrid type with one concentration's PDE and the other's anlaytical solution was implemented:
- CgPINN_WithCsAnalytic tries to fit Cg's PDE using Cs's analytical solution
- CsPINN_WithCgAnalytic tries to fit Cs's PDE using Cg's analytical solution

Finally, real PINN was implemented to try to solve this exercise:
- AdsorptionPINN tries to solve the exercise with a typical PINN 
- AdsorptionDoublePINN tries to solve it in an original manner with 2 distinct networks, one for each concentration, which are connected to one another during training. 

The result is far from successful but intersting nonetheless.