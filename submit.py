import subprocess

cc = 0
for mom in [0.05, 0.1, 0.2, 0.4, 0.6]:
    for lam in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        base_cmd = f'sbatch --job-name=DM_{lam}_{mom} --gres=gpu:1 --mem=16G --cpus-per-task=8 --ntasks=1 --partition=gpu --output=run_logs/rerun_dist_{lam}_{mom}.log --error=run_logs/rerun_dist_{lam}_{mom}.err'
        cmd = base_cmd.split(' ')
        cmd.append(f'--wrap=/bin/bash -c "source /home/yyang/miniconda3/bin/activate; CUDA_VISIBLE_DEVICES={int(cc % 2)} python 30_train_norm.py --lam {lam} --mo {mom}; CUDA_VISIBLE_DEVICES={int(cc % 2)} python 31_eval.py --lam {lam} --mo {mom}"')
        print(cmd)
        subprocess.run(cmd)
        cc +=1
        # cuda_visible_devices=$1 python 04_train_spl.py -m $2; cuda_visible_devices=$1 python 03_eval.py -m $2
for mom in [0.8, 0.9]:
    for lam in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        base_cmd = f'sbatch --job-name=DM_{lam}_{mom} --gres=gpu:1 --mem=16G --cpus-per-task=8 --ntasks=1 --partition=gpu_v100 --output=run_logs/rerun_dist_{lam}_{mom}.log --error=run_logs/rerun_dist_{lam}_{mom}.err'
        cmd = base_cmd.split(' ')
        cmd.append(f'--wrap=/bin/bash -c "source /home/yyang/miniconda3/bin/activate; CUDA_VISIBLE_DEVICES=1 python 30_train_norm.py --lam {lam} --mo {mom}; CUDA_VISIBLE_DEVICES=1 python 31_eval.py --lam {lam} --mo {mom}"')
        print(cmd)
        subprocess.run(cmd)