#!/bin/bash

CODE_PATH="/home/icb/felix.drost/code/Mutagenesis"
OUT_PATH_BASE="/home/icb/felix.drost/code/Mutagenesis/logs/across/"
PARTITION="cpu_p"

N_EXP="1"
START_IDX=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

N="8"
M="10"

for si in ${START_IDX[@]}; do
      sleep 0.1
      job_file="${OUT_PATH_BASE}/job_files/greedy_run_${si}.cmd"
      echo "#!/bin/bash
#SBATCH -J ${si}mgALGB
#SBATCH -o ${OUT_PATH_BASE}/AL_greedy_${si}.out
#SBATCH -e ${OUT_PATH_BASE}/AL_greedy_${si}.err
#SBATCH -p ${PARTITION}
#SBATCH -t 2-00:00:00
#SBATCH -c 6
#SBATCH --mem=5G
#SBATCH --exclude=icb-gpusrv01,icb-gpusrv02
#SBATCH --nice=10000
source ~/.bash_profile
conda activate mutagenesis
python3 ${CODE_PATH}/greedy_across_tcr.py --start_idx ${si} --n_exp ${N_EXP} --n ${N} --m ${M}
" > ${job_file}
        sbatch $job_file
done
