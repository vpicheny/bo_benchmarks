This run was done from scratch using following settings in `run.sh`:
```
LEVEL=2
DEGREE=3
SHOCK=2.
CFL=0.5
```
It has run for 95 iterations (32 initial steps plus 63 BO steps) before evaluation of the constraint failed on step 64. It was also observed that the too many evaluations do not respect the constraint. 
