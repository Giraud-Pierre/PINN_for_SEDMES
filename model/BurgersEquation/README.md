# PINN on matlab: Model

This folder contain the actual building, training and testing of the PINNs regarding Burger's Equation.

At first, it was tried to fit the analytical solution of this exercise using a neural network with a PINN like structure:
- BurgersPINNAnalyticalSol

Then, real PINN was implemented to try to solve this exercise:
- BurgersPINN

The result was very conclusive, which was not surprising because others have already made it. It is proof than the model can work but it is much easier with only one variable instead of 2 like in the adsorption problem.