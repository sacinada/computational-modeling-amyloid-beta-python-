## Computational Modeling of Amyloid-β Dynamics

This project was developed as part of the Computational Mathematical Biology (CMP) program.
It implements a simplified dynamical model of amyloid-β production, secretion, aggregation,
and clearance, inspired by systems biology models of Alzheimer’s disease.

## Model
The system is described by a set of ordinary differential equations (ODEs) representing:
- intracellular amyloid-β
- extracellular amyloid-β
- aggregated amyloid species

Numerical integration is performed using SciPy (odeint, solve_ivp).

## Extensions
The project also explores the effect of a pharmacological intervention (Memantine)
on intracellular amyloid-β dynamics by modifying production and degradation terms.

## Tools
Python, NumPy, SciPy, Matplotlib
