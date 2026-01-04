# Fundamentals of Numerical Programming I - heiAIMS course

This is the lecture material for the first 
week of the heiAIMS "Numerical Programming" course.

The course covers both theoretical foundations and 
practical implementations in Python.

> [!TIP]
> You can directly jump into the script [here](/source/script/loux/main.pdf).

## Course Overview

1. Digital Representation of Numbers ([📝 exercise sheet](/source/sheets/sheet1/loux/main.pdf))
2. Solving Linear Systems ([📝 exercise sheet](/source/sheets/sheet2/loux/main.pdf))
3. Numerical Derivatives and Interpolation ([📝 exercise sheet](/source/sheets/sheet3/loux/main.pdf))
4. Optimization ([📝 exercise sheet](/source/sheets/sheet4/loux/main.pdf))
5. Numerical Integration ([📝 exercise sheet](/source/sheets/sheet5/loux/main.pdf))

The second half of the course will cover numerical
techniques for ordinary differential equations. Other 
important topics not covered include spectral methods 
(in particular the Fast Fourier Transform algorithm
by Cooley and Tukey), and partial differential equations.

## Python Environment

All programming required for the course can be done online
in [Google Colab](https://colab.research.google.com/). You may
also use your development environment of choice.

## Exercises

- sheet 1: essentially FOSM sheet 1 + some of my examples
- sheet 2: implement triangular solve for diffusion, linear regression task, show that matrix is invertible (?), iterative methods(?), compare interpolation orders
- sheet 3: check order of convergence for different finite difference schemes, application: PDE solve (take-away: we can
already do a lot with finite differences)
- sheet 4: gradient descent, newton order comparison, problem of local minima, implicit methods for finite differencing
- sheet 5: numerical integration comparison, Simpson - MC comparison