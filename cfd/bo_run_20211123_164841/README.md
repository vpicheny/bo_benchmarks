This run was done from scratch using following settings in `run.sh`:
```
LEVEL=2
DEGREE=3
SHOCK=2.
CFL=0.5
```

In this run we fix an issue with initial variance for the constraint model. Unfortunately it still does not quite solve the issue of many evaluations not respecting the constraint. In addition, the model inversion failed at BO step 66 due to numerical instability.
