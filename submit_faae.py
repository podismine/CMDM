import subprocess

cc = 0
commbines = [
    [0.1, 5],
    [0.5, 5],
    [0.01, 20],
    [0.5, 1],
]
rvals = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]

for comb in commbines:
    for r in rvals:
        alpha = comb[0]
        gamma = comb[1]
        
        base_cmd = f'sbatch --job-name=FAAE_{alpha}_{gamma}_{r} --gres=gpu:1 --mem=8G --cpus-per-task=4 --ntasks=1 --partition=gpu --output=run_logs/FAAE_{alpha}_{gamma}_{r}.log --error=run_logs/FAAE_{alpha}_{gamma}_{r}.err'
        cmd = base_cmd.split(' ')
        cmd.append(f'--wrap=/bin/bash -c "source /home/yyang/miniconda3/bin/activate; CUDA_VISIBLE_DEVICES={int(cc % 2)} python 15_train_favae.py -a {alpha} -g {gamma} -r {r}; CUDA_VISIBLE_DEVICES={int(cc % 2)} python 16_eval_cvae.py -a {alpha} -g {gamma} -r {r}"')
        print(cmd)
        subprocess.run(cmd)
        cc +=1
        # cuda_visible_devices=$1 python 04_train_spl.py -m $2; cuda_visible_devices=$1 python 03_eval.py -m $2
