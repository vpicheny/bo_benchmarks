############################################################################

library(DiceKriging)
library(DiceOptim)
library(lhs)

# We set seed for reproductivity
set.seed(1)

############################################################################
# My cost function
############################################################################
myOptimFun<-function(x)
{
  # Write params in design_vector_0.dat
  write(length(x), file = "design_vector_0.dat", sep="\n")
  for (i in 1:length(x)) {
    write(x[i], file = "design_vector_0.dat", sep="\n", append=TRUE)
  }
  
  # Workflow for Igloo
  system("./run.sh")
  
  # Retrieve the cost value in simulation_result_0.dat
  f = system("tail -n 1 simulation_result_0.dat",  intern = TRUE)
  
  return(as.numeric(f))
}

############################################################################
# My constraint
############################################################################
myOptimConst<-function(x)
{
  # Retrieve the cost value in simulation_result_1.dat
  f = system("tail -n 1 simulation_result_1.dat",  intern = TRUE)
  
  return(as.numeric(f))
}
