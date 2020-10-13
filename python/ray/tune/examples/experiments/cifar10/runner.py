import subprocess


for n in [500, 7000]:
    for ucb in [True, False]:
        params = {
            "-n_experiments": f'{n}',
            "-n_workers": '10',
            "-perturbation_interval": '5',
            "-training_iteration": '1000',
        }
        script = ["python", "exercise_cifar10.py"]

        for k in params.keys():
            script.append(k)
            script.append(params[k])

        script.append('-save_dir')
        save_dir_name = 'data/'
        for k in params.keys():
            save_dir_name += f"{k}_{params[k]}"

            if ucb:
                save_dir_name += '-ucb'

        script.append(save_dir_name)

        if ucb:
            script.append('-ucb')

        print(f'run command : {script}')

        subprocess.call(script, shell=True)
