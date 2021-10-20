#!/bin/bash

CODE_PATH="/home/icb/felix.drost/code/Mutagenesis"
OUT_PATH_BASE="/home/icb/felix.drost/code/Mutagenesis/logs"
PARTITION="cpu_p"

N_EXP="10"
START_IDX=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
TEST_MODE="False"

N="16"
M="5"

for si in ${START_IDX[@]}; do
      sleep 0.1
      job_file="${OUT_PATH}/jobs/run_${MODEL_CLASS}_${DATA_SET}_${x1}_${x2}_${GS_KEY}.cmd"
      echo "#!/bin/bash
#SBATCH -J ${START_IDX}mgALGB
#SBATCH -o ${OUT_PATH_BASE}/AL_greedy_${START_IDX}.out
#SBATCH -e ${OUT_PATH_BASE}/AL_greedy_${START_IDX}.err
#SBATCH -p ${PARTITION}
#SBATCH -t 2-00:00:00
#SBATCH -c 6
#SBATCH --mem=5G
#SBATCH --exclude=icb-gpusrv01,icb-gpusrv02
#SBATCH --nice=10000
source ~/.bash_profile
conda activate mutagenesis
python3 ${CODE_PATH}/greedy_baseline_within_tcr.py --start_idx ${si} --test_mode ${TEST_MODE} --n_exp ${n_exp} --n ${N} --m ${M}
" > ${job_file}
        sbatch $job_file
done
