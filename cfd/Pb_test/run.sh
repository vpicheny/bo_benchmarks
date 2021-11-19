#!/bin/bash

help()
{
    echo
    echo "Single run of a CFD simulation"
    echo
    echo "Syntax: run.sh -i|--igloo-path <path> -d|--data-dir <path>"
    echo "options:"
    echo "i, igloo-path      Path to igloo bin directory"
    echo "d, data-dir        Path to directory for data files."
    echo "h, help            Print this message."
    echo
}

# parse input arguments
# source: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    -h|--help)
        help
        exit 0
        ;;
    -i|--igloo-path)
        IGLOOPATH="$2"
        shift # past argument
        shift # past value
        ;;
    -d|--data-dir)
        DATA_DIR="$2"
        shift # past argument
        shift # past value
        ;;
    *)
        echo "Unknown parameter passed: $1"
        help
        exit 1
    esac
done

if [ -z "$IGLOOPATH" ]; then
    echo "Path to Igloo executables is not set"
    help
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "Path to directory for data is not set, assuming current directory"
    DATA_DIR="."
fi

cp baseline.dat $DATA_DIR
cp box.dat $DATA_DIR

cd $DATA_DIR  # current path: $DATA_DIR

############ get design parameters from file #######

U1=`cat design_vector_0.dat | awk 'NR==2'`
U2=`cat design_vector_0.dat | awk 'NR==3'`
U3=`cat design_vector_0.dat | awk 'NR==4'`
U4=`cat design_vector_0.dat | awk 'NR==5'`
U5=`cat design_vector_0.dat | awk 'NR==6'`
U6=`cat design_vector_0.dat | awk 'NR==7'`
U7=`cat design_vector_0.dat | awk 'NR==8'`
U8=`cat design_vector_0.dat | awk 'NR==9'`


D1=`cat design_vector_0.dat | awk 'NR==10'`
D2=`cat design_vector_0.dat | awk 'NR==11'`
D3=`cat design_vector_0.dat | awk 'NR==12'`
D4=`cat design_vector_0.dat | awk 'NR==13'`
D5=`cat design_vector_0.dat | awk 'NR==14'`
D6=`cat design_vector_0.dat | awk 'NR==15'`
D7=`cat design_vector_0.dat | awk 'NR==16'`
D8=`cat design_vector_0.dat | awk 'NR==17'`

############ set simulation parameters #######

#IGLOOPATH='~/projects/igloo_cfd/igloo/build/bin'
PROCS=16

# default settings
# LEVEL=2  # between 0 and 5
# DEGREE=3 # between 3 and 5
# SHOCK=1. # between 0. and 5.
# CFL=0.9  # between 0. and 1.

# settings that are not as useful but eliminate crashes
LEVEL=2
DEGREE=3
SHOCK=2.
CFL=0.5

############ other dependent parameters #######

GAUSS=`echo print $DEGREE + 1 | perl`

########### run simulation  #############

rm -rf Eval
mkdir Eval
cd Eval  # current path: $DATA_DIR/Eval

cp ../baseline.dat .
cp ../box.dat .

$IGLOOPATH/igGenCurve -case airfoil_16 -up1 $U1 -up2 $U2 -up3 $U3 -up4 $U4 -up5 $U5 -up6 $U6 -up7 $U7 -up8 $U8 -down1 $D1 -down2 $D2 -down3 $D3 -down4 $D4 -down5 $D5 -down6 $D6 -down7 $D7 -down8 $D8
$IGLOOPATH/igGenMesh -case ns_multi_patch -mesh mesh.dat -n1 16 -degree 3 -gauss 4
$IGLOOPATH/igPreRefiner -mesh mesh.dat -box ./box.dat
$IGLOOPATH/igPreRefiner -mesh mesh_refined.dat -wall 2 -no_hanging
$IGLOOPATH/igPreElevator -mesh mesh_refined.dat -degree $DEGREE -gauss $GAUSS 
$IGLOOPATH/igMeshDeform -mesh mesh_elevated.dat -rotation -alpha -1.
$IGLOOPATH/igGenSol -case ns_multi_patch -mesh mesh_deformed.dat -initial solution.dat -mach 0.7
$IGLOOPATH/igPartit -mesh mesh_deformed.dat -npart1 $PROCS

mpirun -np $PROCS $IGLOOPATH/igloo -solver navier-stokes -mesh mesh_distributed.dat -initial solution.dat -cfl $CFL -time 20.0 -save_period 5. -mach 0.7 -reynolds 1.E15 -shock $SHOCK -viscosity_smoothing 3 -integrator rk2 -refine_coef 1.5 -coarsen_coef 0.5 -refine_max $LEVEL -adapt_period 10 -adapt_xmax 3. -adapt_xmin -1. -adapt_ymax 1. -adapt_ymin -1. > igloo.log

############# get results from file ##########

DRAG=`tail -1 ./efforts.dat | awk ' { print $2 } '`
LIFT=`tail -1 ./efforts.dat | awk ' { print $3 } '`
MOM=`tail -1 ./efforts.dat | awk ' { print $6 } '`

########### check if simulation crashed #####

DRAGFLOAT=`echo print $DRAG | perl`

isnum() { local ck=${1#[+-]};ck=${ck/.};[ "$ck" ]&&[ -z "${ck//[0-9]}" ];}

if isnum "$DRAGFLOAT"
   then   SIMFLAG=0
   else   SIMFLAG=1
fi

########## compute cost and constraint values ########

LREF=0.125

FUNCT=`echo print $DRAG | perl`
CONST=`echo print $LREF - $LIFT | perl`

######## write file for optimizer #################

echo "1" >> simulation_result_0.dat
echo $SIMFLAG >> simulation_result_0.dat
echo $FUNCT >> simulation_result_0.dat

echo "1" >> simulation_result_1.dat
echo $SIMFLAG >> simulation_result_1.dat
echo $CONST >> simulation_result_1.dat

mv simulation_result_0.dat ../
mv simulation_result_1.dat ../
cd ../  # current path: $DATA_DIR

################# save data ############

if [ -e optim.dat ]
then
    LASTID=`tail -1 ./optim.dat | awk ' { print $1 } '`
    ID=$(($LASTID+1))
else
    ID=0
fi

mv Eval Eval_$ID

echo $ID $FUNCT $CONST $U1 $U2 $U3 $U4 $U5 $U6 $U7 $U8 $D1 $D2 $D3 $D4 $D5 $D6 $D7 $D8>> optim.dat

################ save best data ######

function float_gt() {
    perl -e "{if($1>$2){print 1} else {print 0}}"
}

function float_lt() {
    perl -e "{if($1<$2){print 1} else {print 0}}"
}


if [ -e best.dat ]
then

    BESTID=`tail -1 ./best.dat | awk ' { print $1 } '`
    BESTFUNCT=`tail -1 ./best.dat | awk ' { print $2 } '`
    BESTCONST=`tail -1 ./best.dat | awk ' { print $3 } '`

    BESTU1=`tail -1 ./best.dat | awk ' { print $4 } '`
    BESTU2=`tail -1 ./best.dat | awk ' { print $5 } '`
    BESTU3=`tail -1 ./best.dat | awk ' { print $6 } '`
    BESTU4=`tail -1 ./best.dat | awk ' { print $7 } '`
    BESTU5=`tail -1 ./best.dat | awk ' { print $8 } '`
    BESTU6=`tail -1 ./best.dat | awk ' { print $9 } '`
    BESTU7=`tail -1 ./best.dat | awk ' { print $10 } '`
    BESTU8=`tail -1 ./best.dat | awk ' { print $11 } '`

    BESTD1=`tail -1 ./best.dat | awk ' { print $12 } '`
    BESTD2=`tail -1 ./best.dat | awk ' { print $13 } '`
    BESTD3=`tail -1 ./best.dat | awk ' { print $14 } '`
    BESTD4=`tail -1 ./best.dat | awk ' { print $15 } '`
    BESTD5=`tail -1 ./best.dat | awk ' { print $16 } '`
    BESTD6=`tail -1 ./best.dat | awk ' { print $17 } '`
    BESTD7=`tail -1 ./best.dat | awk ' { print $18 } '`
    BESTD8=`tail -1 ./best.dat | awk ' { print $19 } '`

    if [ $(float_lt $CONST 0) == 1 ] ; then
	if [ $(float_lt $FUNCT $BESTFUNCT) == 1 ] ; then
	    BESTID=$ID
            BESTFUNCT=$FUNCT
	    BESTCONST=$CONST
	    BESTU1=$U1
	    BESTU2=$U2
	    BESTU3=$U3
	    BESTU4=$U4
	    BESTU5=$U5
	    BESTU6=$U6
	    BESTU7=$U7
	    BESTU8=$U8
	    BESTD1=$D1
	    BESTD2=$D2
	    BESTD3=$D3
	    BESTD4=$D4
	    BESTD5=$D5
	    BESTD6=$D6
	    BESTD7=$D7
	    BESTD8=$D8
	fi
    fi

else

    BESTID=$ID
    if [ $(float_lt $CONST 0) == 1 ] ; then
	BESTFUNCT=$FUNCT
    else
	BESTFUNCT=1.E10
    fi
    BESTCONST=$CONST
    BESTU1=$U1
    BESTU2=$U2
    BESTU3=$U3
    BESTU4=$U4
    BESTU5=$U5
    BESTU6=$U6
    BESTU7=$U7
    BESTU8=$U8
    BESTD1=$D1
    BESTD2=$D2
    BESTD3=$D3
    BESTD4=$D4
    BESTD5=$D5
    BESTD6=$D6
    BESTD7=$D7
    BESTD8=$D8

fi

echo $BESTID $BESTFUNCT $BESTCONST $BESTU1 $BESTU2 $BESTU3 $BESTU4 $BESTU5 $BESTU6 $BESTU7 $BESTU8 $BESTD1 $BESTD2 $BESTD3 $BESTD4 $BESTD5 $BESTD6 $BESTD7 $BESTD8 >> best.dat


########### end ##################

exit 0
