import yaml
import os

index_range = range(1, 11)
config_dir = "configs/"

# SOTA on halfcheetah-medium-v2, 67%
# for index in index_range:
#     filename = f"halfcheetah-sota-seed{index}.yaml"
#     config = {
#         "discount2": 0.99,
#         "coef": 1.0,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": index,
#         "T": 1,
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)

#     command = f"python launch/remote_run.py --job_name dac-sota-seed{index} main.py --config {filename} --run"
#     os.system(command)

discount2 = [0.9, 0.99, 1.0]  # 0.999 is the best
coef = [0.5, 0.75, 1.25, 1.5]  # 1.0 is the best
T = [2, 3, 4]  # 1 is the best
compute_consistency = [True, False]  # True is the best
algo = ['bc', 'ql', ]

# best_config = {
#     "discount2": 0.999,
#     "coef": 1.0,
#     "lr_decay": False,
#     "early_stop": False,
#     "seed": 0,
#     "T": 1,
# }

filename = f"halfcheetah-best.yaml"
config = {
    "discount2": 0.999,
    "coef": 1.0,
    "lr_decay": False,
    "early_stop": False,
    "seed": 0,
    "T": 1,
}
filename = os.path.join(config_dir, filename)
with open(filename, "w") as file:
    yaml.dump(config, file)
command = f"python launch/remote_run.py --job_name dac-best main.py --config {filename} --run"
os.system("git pull origin master")
os.system("git add .")
os.system(f"git commit -m '{command}''")
os.system("git push origin master'")
os.system(command)

# discount2
for dis in discount2:
    filename = f"halfcheetah-discount2{dis}.yaml"
    config = {
        "discount2": dis,
        "coef": 1.0,
        "lr_decay": False,
        "early_stop": False,
        "seed": 0,
        "T": 1,
    }
    filename = os.path.join(config_dir, filename)
    with open(filename, "w") as file:
        yaml.dump(config, file)

    command = f"python launch/remote_run.py --job_name dac-discount2{dis} main.py --config {filename} --run"
    os.system("git pull origin master")
    os.system("git add .")
    os.system(f"git commit -m '{command}''")
    os.system("git push origin master'")
    os.system(command)

# coef
for c in coef:
    filename = f"halfcheetah-coef{c}.yaml"
    config = {
        "discount2": 0.999,
        "coef": c,
        "lr_decay": False,
        "early_stop": False,
        "seed": 0,
        "T": 1,
    }
    filename = os.path.join(config_dir, filename)
    with open(filename, "w") as file:
        yaml.dump(config, file)

    command = f"python launch/remote_run.py --job_name dac-coef{c} main.py --config {filename} --run"
    os.system("git pull origin master")
    os.system("git add .")
    os.system(f"git commit -m '{command}''")
    os.system("git push origin master'")
    os.system(command)

# T
for t in T:
    filename = f"halfcheetah-T{t}.yaml"
    config = {
        "discount2": 0.999,
        "coef": 1.0,
        "lr_decay": False,
        "early_stop": False,
        "seed": 0,
        "T": t,
    }
    filename = os.path.join(config_dir, filename)
    with open(filename, "w") as file:
        yaml.dump(config, file)

    command = f"python launch/remote_run.py --job_name dac-t{t} main.py --config {filename} --run"
    os.system("git pull origin master")
    os.system("git add .")
    os.system(f"git commit -m '{command}''")
    os.system("git push origin master'")
    os.system(command)


# compute_consistency
filename = f"halfcheetah-compute_consistency.yaml"
config = {
    "discount2": 0.999,
    "coef": 1.0,
    "lr_decay": False,
    "early_stop": False,
    "seed": 0,
    "T": 1,
    "compute_consistency": False,
}
filename = os.path.join(config_dir, filename)
with open(filename, "w") as file:
    yaml.dump(config, file)
command = f"python launch/remote_run.py --job_name dac-compute_consistency main.py --config {filename} --run"
os.system("git pull origin master")
os.system("git add .")
os.system(f"git commit -m '{command}''")
os.system("git push origin master'")
os.system(command)

for al in algo:
    filename = f"halfcheetah-algo{al}.yaml"
    config = {
        "discount2": 0.999,
        "coef": 1.0,
        "lr_decay": False,
        "early_stop": False,
        "seed": 0,
        "T": 1,
        "algo": al,
    }
    filename = os.path.join(config_dir, filename)   
    with open(filename, "w") as file:
        yaml.dump(config, file)
    command = f"python launch/remote_run.py --job_name dac-algo{al} main.py --config {filename} --run"
    os.system("git pull origin master")
    os.system("git add .")
    os.system(f"git commit -m '{command}''")
    os.system("git push origin master'")
    