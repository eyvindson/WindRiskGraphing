#!/bin/bash -l

#SBATCH -J TEST_OPT
#SBATCH --account=project_2004798
#SBATCH -o error/my_output_%j
#SBATCH -e error/my_output_err_%j
#SBATCH -t 07:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH -p serial 
#SBATCH -N 20
#SBATCH -n 20
#SBATCH --partition=large
#SBATCH --mail-type=END
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kyle.eyvindson@luke.fi


module load geoconda
wait

declare -a arr=(5 10 15 20)


 for i in "${arr[@]}"; do 

srun -J "Low_High"  -N 1 -n 1 python Python/Stochastic_Programming_problem.py --V 0  --P $i --OPT CVAR --INT 0.01 --SCEN 50&
srun -J "Mod_Mid"  -N 1 -n 1 python Python/Stochastic_Programming_problem.py --V 1  --P $i --OPT CVAR --INT 0.01 --SCEN 50&
srun -J "Mod_Low"  -N 1 -n 1 python Python/Stochastic_Programming_problem.py --V 2  --P $i --OPT CVAR --INT 0.01 --SCEN 50&
srun -J "High_Low"  -N 1 -n 1 python Python/Stochastic_Programming_problem.py --V 3  --P $i --OPT CVAR --INT 0.01 --SCEN 50&
srun -J "Low_Low"  -N 1 -n 1 python Python/Stochastic_Programming_problem.py --V 4 --P $i --OPT CVAR --INT 0.01 --SCEN 50&

done

wait

