### settings file with function to optimize, constraints, library loads, ...

source(file="../optim_settings.r")

########### problem dimensions ##################

nVariables <- 16
nPtsDoE    <- 32
nStepEGO   <- 120
myFun      <- myOptimFun
myConst    <- myOptimConst


########## bound constraints (min max) ##############

param     <- matrix(nrow = nVariables, ncol = 2)
param[1,] <- c( 0.02 , 0.05)
param[2,] <- c( 0.03 , 0.08)
param[3,] <- c( 0.03 , 0.10)
param[4,] <- c( 0.03 , 0.10)
param[5,] <- c( 0.03 , 0.10)
param[6,] <- c( 0.02 , 0.08)
param[7,] <- c( 0.01 , 0.07)
param[8,] <- c( 0.005 , 0.05)

param[9,]  <- c( -0.05 , -0.02)
param[10,] <- c( -0.06 , -0.02)
param[11,] <- c( -0.07 , -0.03)
param[12,] <- c( -0.08 , -0.03)
param[13,] <- c( -0.08 , -0.03)
param[14,] <- c( -0.07 , -0.01)
param[15,] <- c( -0.05 , -0.00)
param[16,] <- c( -0.03 , -0.00)




############# DoE generation #####################

lhs <- create_oalhs(nPtsDoE, nVariables, TRUE, FALSE)
doe <- matrix(0, nPtsDoE, nVariables)
for (i in 1:nVariables) {
  doe[, i] <- param[i,1]+lhs[1:nPtsDoE,i]*(param[i,2]-param[i,1])
}

design <- data.frame(doe)


################ function evaluation for DoE ############

response.myFun <- vector(mode = "numeric", length = nPtsDoE)
response.myConst <- vector(mode = "numeric", length = nPtsDoE)

for(i in 1:nPtsDoE) {

    x <- doe[i,]
    response.myFun[i] <- myFun(x)
    response.myConst[i] <- myConst(x)

}

response.myConst <- data.frame(response.myConst)
names(response.myConst) <- "y"

response.myFun <- data.frame(response.myFun)
names(response.myFun) <- "y"

save.image(file = "data_preEGO.RData")


################ construct kriging model ##################

model.myFun <- km(~1, design=design, response=response.myFun)
model.myConst <- km(~1, design=design, response=response.myConst)

save.image(file = "data_preEGO.RData")


################# EGO algorithm ########################

lower  <- c(param[,1])
upper  <- c(param[,2])
myOptim.res <- matrix(0, nStepEGO, nVariables+2)

oEGO <- EGO.cst(model.fun = model.myFun, fun = myFun, model.constraint = model.myConst,
     	crit = "EFI", constraint = myConst, equality = FALSE, lower = lower,
	upper = upper, nsteps = 1, optimcontrol = list(method = "genoud", maxit = 20))

myOptim.res[1,] <- c(oEGO$par,oEGO$value,oEGO$constraint)

save.image(file = paste("data_iteration_",1,".RData",sep=""))

for (i in seq(2,nStepEGO,length=nStepEGO-1)) {

    oEGO <- EGO.cst(model.fun = oEGO$lastmodel.fun, fun = myFun, model.constraint = oEGO$lastmodel.constraint,
    	    crit = "EFI", constraint = myConst, equality = FALSE, lower = lower,
    	    upper = upper, nsteps = 1, optimcontrol = list(method = "genoud", maxit = 20))

   myOptim.res[i,] <- c(oEGO$par,oEGO$value,oEGO$constraint)
   save.image(file = paste("data_iteration_",i,".RData",sep=""))
 }
