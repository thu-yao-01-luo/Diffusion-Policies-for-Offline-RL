import yaml
import os

index_range = range(1, 11)
config_dir = "configs/"

# T=1
for index in index_range:
    filename = f"halfcheetah-sota-seed{index}.yaml"
    config = {
        "discount2": 0.99,
        "coef": 1.0,
        "lr_decay": False,
        "early_stop": False,
        "seed": index,
        "T": 1,
    }
    filename = os.path.join(config_dir, filename)
    with open(filename, "w") as file:
        yaml.dump(config, file)

    command = f"python launch/remote_run.py --job_name dac-sota-seed{index} main.py --config {filename} --run"
    os.system(command)
