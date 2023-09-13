from multiprocessing import Process
import yaml
import os
import random
import time

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
# coef = [0.5, 0.75, 1.25, 1.5]  # 1.0 is the best
coef = [1.25, 1.5]
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

# filename = f"halfcheetah-best.yaml"
# config = {
#     "discount2": 0.999,
#     "coef": 1.0,
#     "lr_decay": False,
#     "early_stop": False,
#     "seed": 0,
#     "T": 1,
# }
# filename = os.path.join(config_dir, filename)
# with open(filename, "w") as file:
#     yaml.dump(config, file)
# command = f"python launch/remote_run.py --job_name dac-best main.py --config {filename} --run"
# os.system("git pull origin master")
# os.system("git add .")
# os.system(f"git commit -m '{command}''")
# os.system("git push origin master'")
# os.system(command)

# # discount2
# for dis in discount2:
#     filename = f"halfcheetah-discount2{dis}.yaml"
#     config = {
#         "discount2": dis,
#         "coef": 1.0,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": 0,
#         "T": 1,
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)

#     command = f"python launch/remote_run.py --job_name dac-discount2{dis} main.py --config {filename} --run"
#     os.system("git pull origin master")
#     os.system("git add .")
#     os.system(f"git commit -m '{command}''")
#     os.system("git push origin master'")
#     os.system(command)

# coef
# for c in coef:
#     filename = f"halfcheetah-coef{c}.yaml"
#     config = {
#         "discount2": 0.999,
#         "coef": c,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": 0,
#         "T": 1,
#         "device": random.randint(0, 7)
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)

#     command = f"python launch/remote_run.py --job_name dac-coef{c} main.py --config {filename} --run"
#     os.system("git pull origin master")
#     os.system("git add .")
#     os.system(f"git commit -m '{command}''")
#     os.system("git push origin master'")
#     os.system(command)

# # T
# for t in T:
#     filename = f"halfcheetah-T{t}.yaml"
#     config = {
#         "discount2": 0.999,
#         "coef": 1.0,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": 0,
#         "T": t,
#         "device": random.randint(0, 7)
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)

#     command = f"python launch/remote_run.py --job_name dac-t{t} main.py --config {filename} --run"
#     os.system("git pull origin master")
#     os.system("git add .")
#     os.system(f"git commit -m '{command}''")
#     os.system("git push origin master'")
#     os.system(command)


# # compute_consistency
# filename = f"halfcheetah-compute_consistency.yaml"
# config = {
#     "discount2": 0.999,
#     "coef": 1.0,
#     "lr_decay": False,
#     "early_stop": False,
#     "seed": 0,
#     "T": 1,
#     "compute_consistency": False,
#     "device": random.randint(0, 7)
# }
# filename = os.path.join(config_dir, filename)
# with open(filename, "w") as file:
#     yaml.dump(config, file)
# command = f"python launch/remote_run.py --job_name dac-compute_consistency main.py --config {filename} --run"
# os.system("git pull origin master")
# os.system("git add .")
# os.system(f"git commit -m '{command}''")
# os.system("git push origin master'")
# os.system(command)

# for al in algo:
#     filename = f"halfcheetah-algo{al}.yaml"
#     config = {
#         "discount2": 0.999,
#         "coef": 1.0,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": 0,
#         "T": 1,
#         "algo": al,
#         "device": random.randint(0, 7)
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)
#     command = f"python launch/remote_run.py --job_name dac-algo{al} main.py --config {filename} --run"
#     os.system("git pull origin master")
#     os.system("git add .")
#     os.system(f"git commit -m '{command}''")
#     os.system("git push origin master'")

# for c in coef:
#     filename = f"halfcheetah-coef{c}.yaml"
#     config = {
#         "discount2": 0.999,
#         "coef": c,
#         "lr_decay": False,
#         "early_stop": False,
#         "seed": 0,
#         "T": 1,
#         "device": random.randint(0, 7)
#     }
#     filename = os.path.join(config_dir, filename)
#     with open(filename, "w") as file:
#         yaml.dump(config, file)

#     command = f"python main.py --config {filename}"
#     os.system("git pull origin master")
#     os.system("git add .")
#     os.system(f"git commit -m '{command}''")
#     os.system("git push origin master'")
#     os.system(command)


def run_python_file(job, filename, main="main.py"):
    command = f"python launch/remote_run.py --job_name {job} {main} --config {filename} --run"
    os.system("git add .")
    os.system(f"git commit -m '{command}''")
    os.system("git pull origin master")
    os.system("git push origin master'")
    os.system(command)


def make_config_file(filename, config):
    with open(os.path.join(filename), "w") as file:
        yaml.dump(config, file)


# if __name__ == "__main__":
def jun22_experiment1():
    file_paths = []
    for t in T:
        filename = f"halfcheetah-T{t}.yaml"
        config = {
            "discount2": 0.999,
            "coef": 1.0,
            "lr_decay": False,
            "early_stop": False,
            "seed": 0,
            "T": t,
            "device": random.randint(0, 7)
        }
        filename = os.path.join(config_dir, filename)
        file_paths.append(filename)
        with open(filename, "w") as file:
            yaml.dump(config, file)
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
        "device": random.randint(0, 7)
    }
    filename = os.path.join(config_dir, filename)
    file_paths.append(filename)
    with open(filename, "w") as file:
        yaml.dump(config, file)

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
            "device": random.randint(0, 7)
        }
        filename = os.path.join(config_dir, filename)
        file_paths.append(filename)
        with open(filename, "w") as file:
            yaml.dump(config, file)

    env_name = ["hopper-medium-v2", "walker2d-medium-v2", "antmaze-umaze-v0"]

    for env in env_name:
        filename = f"{env}-first-try.yaml"
        config = {
            "discount2": 0.999,
            "coef": 1.0,
            "lr_decay": False,
            "early_stop": False,
            "seed": 0,
            "T": 1,
            "algo": "dac",
            "env_name": env,
            "device": random.randint(0, 7)
        }
        filename = os.path.join(config_dir, filename)
        file_paths.append(filename)
        with open(filename, "w") as file:
            yaml.dump(config, file)

    iql = ["expectile", "quantile", "exponential"]

    for iql_style in iql:
        filename = f"halfcheetah-iql{iql_style}.yaml"
        config = {
            "discount2": 1.0,
            "coef": 1.0,
            "lr_decay": False,
            "early_stop": False,
            "seed": 0,
            "T": 1,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "iql_style": iql_style,
        }
        filename = os.path.join(config_dir, filename)
        file_paths.append(filename)
        with open(filename, "w") as file:
            yaml.dump(config, file)


def jun22_iql():
    file_paths = []
    iql = ["expectile", "quantile", "exponential"]
    config_dir = "configs/"
    for iql_style in iql:
        filename = f"halfcheetah-iql{iql_style}.yaml"
        config = {
            "discount2": 1.0,
            "coef": 1.0,
            "lr_decay": False,
            "early_stop": False,
            "seed": 0,
            "T": 1,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "iql_style": iql_style,
        }
        filename = os.path.join(config_dir, filename)
        file_paths.append(filename)
    for filename in file_paths:
        run_python_file(filename, filename)

def jun22_all_env():  # check the effect in different envs, with different seeds
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "antmaze-umaze-v0"]
    algo = ["bc", "ql", "dac"]
    seeds = [11, 12, 13]
    config_dir = "configs/"
    for env_name in env:
        for al in algo:
            for seed in seeds:
                file_name = f"test-{env_name}-{al}-{seed}.yaml"
                config = {
                    "discount2": 0.999,
                    "coef": 1.0,
                    "lr_decay": False,
                    "early_stop": False,
                    "seed": seed,
                    "T": 1,
                    "algo": al,
                    "env_name": env_name,
                    "iql_style": "expectile",
                }
                job_list.append(f"{env_name}-{al}-{seed}")
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun23_discount_all_env():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2"]
    algo = ["dac"]
    seeds = [11, 12, 13]
    config_dir = "configs/"
    for env_name in env:
        for al in algo:
            for seed in seeds:
                file_name = f"test-discount-{env_name}-{al}-{seed}.yaml"
                config = {
                    "discount2": 0.999,
                    "coef": 1.0,
                    "lr_decay": False,
                    "early_stop": False,
                    "seed": seed,
                    "T": 1,
                    "algo": al,
                    "env_name": env_name,
                    "iql_style": "discount",
                }
                job_list.append(f"discount-{env_name}-{al}-{seed}")
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun23_bc_discount():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    algo = ["dac"]
    bc_weights = [3.0, 5.0]
    seeds = [11, 12, 13]
    config_dir = "configs/"
    for env_name in env:
        for al in algo:
            for seed in seeds:
                for bc_weight in bc_weights:
                    file_name = f"bc-weight{bc_weight}-discount-{env_name}-{al}-{seed}.yaml"
                    config = {
                        "discount2": 0.999,
                        "coef": 1.0,
                        "lr_decay": False,
                        "early_stop": False,
                        "seed": seed,
                        "T": 1,
                        "algo": al,
                        "env_name": env_name,
                        "iql_style": "discount",
                        "bc_weight": bc_weight,
                    }
                    job_list.append(
                        f"bcw{bc_weight}-discount-{env_name}-{al}-{seed}")
                    filename = os.path.join(config_dir, file_name)
                    file_paths.append(filename)
                    # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun24_bc_weight():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    bc_weights = [1.0, 1.5, 2.5, 7.5]
    # bc_tunes = [True, False]
    bc_tunes = [True]
    config_dir = "configs/bc_weight/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        job_id = f"{env_name[:6]}-bc"
        file_name = job_id + ".yaml"
        config = {
            "discount2": 0.999,
            "coef": 1.0,
            "seed": 0,
            "T": 1,
            "algo": "bc",
            "env_name": env_name,
            "iql_style": "discount",
            "name": job_id,
            "id": job_id,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        # make_config_file(filename, config)
    for env_name in env:
        for bc_tune in bc_tunes:
            for bc_weight in bc_weights:
                job_id = f"{env_name[:6]}-tune{int(bc_tune)}-bcw{bc_weight}"
                file_name = job_id + ".yaml"
                config = {
                    "discount2": 0.999,
                    "coef": 1.0,
                    "seed": 0,
                    "T": 1,
                    "algo": "dac",
                    "env_name": env_name,
                    "iql_style": "discount",
                    "bc_weight": bc_weight,
                    "tune_bc_weight": bc_tune,
                    "name": job_id,
                    "id": job_id,
                    "std_threshold": 1e-4,
                    "bc_lower_bound": 1e-3,
                    "bc_decay": 0.995,
                }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun24_bc_weight_decay():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    bc_weights = [1.5, 2.5, 5.0, 10.0]
    config_dir = "configs/bc_decay/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        job_id = f"{env_name[:6]}-bc"
        file_name = job_id + ".yaml"
        config = {
            "discount2": 0.999,
            "coef": 1.0,
            "seed": 0,
            "T": 1,
            "algo": "bc",
            "env_name": env_name,
            "iql_style": "discount",
            "name": job_id,
            "id": job_id,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        # make_config_file(filename, config)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-bcw{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "iql_style": "discount",
                "bc_weight": bc_weight,
                "tune_bc_weight": True,
                "name": job_id,
                "id": job_id,
                "std_threshold": 1e-4,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun25_bc_weight():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    bc_weights = [1.5, 2.5, 5.0, 10.0, 20.0]
    config_dir = "configs/bc_control/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-bcc{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "iql_style": "discount",
                "bc_weight": bc_weight,
                "tune_bc_weight": True,
                "name": job_id,
                "id": job_id,
                "std_threshold": 1e-4,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 3.6e-4 if env_name == "hopper-medium-v2" else 2.5e-4,
                "bc_upper_bound": 1e2,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun25_consistency():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    bc_weights = [1.5, 2.5, 10.0, 20.0]
    config_dir = "configs/bc_control_diff_target/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-bcc-high-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "iql_style": "discount",
                "bc_weight": bc_weight,
                "tune_bc_weight": True,
                "name": job_id,
                "id": job_id,
                "std_threshold": 1e-4,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 3.8e-4 if env_name == "hopper-medium-v2" else 2.8e-4,
                "bc_upper_bound": 1e2,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun26_consistency():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    config_dir = "configs/bc_control_consistency/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        job_id = f"{env_name[:6]}-consist"
        file_name = job_id + ".yaml"
        config = {
            "discount2": 0.999,
            "coef": 1.0,
            "seed": 0,
            "T": 1,
            "algo": "dac",
            "env_name": env_name,
            "iql_style": "discount",
            "bc_weight": 10,
            "tune_bc_weight": True,
            "name": job_id,
            "id": job_id,
            "std_threshold": 1e-4,
            "bc_lower_bound": 1e-2,
            "bc_decay": 0.995,
            "value_threshold": 3.8e-4 if env_name == "hopper-medium-v2" else 2.8e-4,
            "bc_upper_bound": 1e2,
            "consistency": False,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun26_consistency_ql():
    file_paths = []
    job_list = []
    bc_weights = [1.5, 2.5, 10.0]
    env = ["walker2d-medium-v2", "halfcheetah-medium-v2"]
    config_dir = "configs/bc_control_ql/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-ql-bcc{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": True,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun26_vae_ac():
    file_paths = []
    job_list = []
    bc_weights = [2.5, 7.5]
    env = ["walker2d-medium-v2", "halfcheetah-medium-v2"]
    config_dir = "configs/vae_ac/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-vae-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "vae-ac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun26_noise_decay():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    scales = [1e-4, 1e-6, 0.0]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/scale/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for scale in scales:
            job_id = f"{env_name[:6]}-scale-{scale}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 7.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "scale": scale,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    # for ind, job in enumerate(job_list):
    #     run_python_file(job, file_paths[ind])


def jun26_bc_weight():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    bc_weights = [1.5, 2.5, 7.5]
    config_dir = "configs/bc_weight/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-tune{int(False)}-bcw{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "iql_style": "discount",
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "std_threshold": 1e-4,
                "bc_lower_bound": 1e-3,
                "bc_decay": 0.995,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    # for ind, job in enumerate(job_list):
    #     run_python_file(job, file_paths[ind])


def jun26_bc():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4]
    config_dir = "configs/bc_weight/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-bc-time{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "bc",
                "env_name": env_name,
                "iql_style": "discount",
                "bc_weight": 2.5,
                "tune_bc_weight": True,
                "name": job_id,
                "id": job_id,
                "std_threshold": 1e-4,
                "bc_lower_bound": 1e-3,
                "bc_decay": 0.99,
                "value_threshold": 2.8e-4,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun27_init_noise_decay():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    scales = [1e-4, 1e-6, 0.0]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/init-noise/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for scale in scales:
            job_id = f"{env_name[:6]}-init-noise-{scale}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 7.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "scale": scale,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun27_ql_noise():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    scales = [1e-4, 1e-6, 0.0]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/ql-noise/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for scale in scales:
            job_id = f"{env_name[:6]}-ql-noise-{scale}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 7.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "scale": scale,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun27_init_noise_decay_fix():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    scales = [1e-4, 1e-6, 0.0]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/init-noise-fix/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for scale in scales:
            job_id = f"{env_name[:6]}-init-noise-fix-{scale}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 7.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "scale": scale,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_sota():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    # scales = [1e-4, 1e-6, 0.0]
    bc_weights = [1.5, 2.5, 7.5]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/sota/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for scale in scales:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-dac-bc-loss-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)

    for env_name in env:
        # for scale in scales:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-dql-bc-loss-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_consist():
    file_paths = []
    job_list = []
    bc_weights = [1.5, 2.5, 7.5]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/no-consist/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for scale in scales:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-no-consist-dac-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_bct():
    file_paths = []
    job_list = []
    # scales = [1e-1, 1e-2, 1e-3]
    # scales = [1e-4, 1e-6, 0.0]
    # bc_weights = [1.5, 2.5, 7.5]
    Ts = [1, 2, 3, 4, 8, 16]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/bct/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for scale in scales:
        # for bc_weight in bc_weights:
        for t in Ts:
            job_id = f"{env_name[:6]}-bct-{t}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": t,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 7.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_sota_noise():
    file_paths = []
    job_list = []
    bc_weights = [1.5, 2.5]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/sota-noise/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for scale in scales:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-dac-bc-loss-noise-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)

    for env_name in env:
        # for scale in scales:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-ql-bc-loss-noise-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_sample_bcw():
    file_paths = []
    job_list = []
    bc_weights = [0.0097, 0.016]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/sample-bcw/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for bc_weight in bc_weights:
            job_id = f"{env_name[:6]}-ql-sample-bcw-{bc_weight}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_sample_low_weight():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-bc-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_sota_noise_t():
    file_paths = []
    job_list = []
    # bc_weights = [1.5, 2.5]
    Ts = [1, 2, 4, 8]
    env = ["halfcheetah-medium-v2"]
    config_dir = "configs/sota-noise-t/"
    os.makedirs(config_dir, exist_ok=True)
    # for env_name in env:
    #     # for scale in scales:
    #     for bc_weight in bc_weights:
    #         job_id = f"{env_name[:6]}-dac-bc-loss-noise-{bc_weight}"
    #         file_name = job_id + ".yaml"
    #         config = {
    #             "discount2": 0.999,
    #             "coef": 1.0,
    #             "seed": 0,
    #             "T": 1,
    #             "algo": "dac",
    #             "env_name": env_name,
    #             "bc_weight": bc_weight,
    #             "tune_bc_weight": False,
    #             "name": job_id,
    #             "id": job_id,
    #             "bc_lower_bound": 1e-2,
    #             "bc_decay": 0.995,
    #             "value_threshold": 2.8e-4,
    #             "bc_upper_bound": 1e2,
    #             "predict_epsilon": True,
    #         }
    #         job_list.append(
    #             job_id)
    #         filename = os.path.join(config_dir, file_name)
    #         file_paths.append(filename)
    #         make_config_file(filename, config)

    for env_name in env:
        # for scale in scales:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-noise-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": 1,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 1.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_walker():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    env = ["walker2d-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-bc-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 5e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun28_hopper():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    # env = ["walker2d-medium-v2"]
    env = ["hopper-medium-v2"]
    Ts = [4, 8]
    config_dir = "configs/sample-bc-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 5e-3,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun29_walker():
    file_paths = []
    job_list = []
    env = ["walker2d-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-bc-low0.1/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-low0.1-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.1,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun29_hopper():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-bc-low-0.1/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-ql-bcw-low0.1-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.1,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun30_walker():
    file_paths = []
    job_list = []
    env = ["walker2d-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-dac-low0.05/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-dac-bcw-low0.05-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.05,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jun30_hopper():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-dac-low-0.5/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for T in Ts:
            job_id = f"{env_name[:6]}-dac-bcw-low0.5-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "bc_weight": 0.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul03_weight():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [2, 8]
    weights = [0.1, 1.0]
    config_dir = "configs/sample-td3-low-0.5/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for weight in weights:
            for T in Ts:
                job_id = f"{env_name[:6]}-td3-bcw{weight}-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "discount2": 0.999,
                    "coef": 1.0,
                    "seed": 0,
                    "T": T,
                    "algo": "td3",
                    "env_name": env_name,
                    "bc_weight": weight,
                    "tune_bc_weight": False,
                    "name": job_id,
                    "id": job_id,
                    "bc_lower_bound": 1e-2,
                    "bc_decay": 0.995,
                    "value_threshold": 2.8e-4,
                    "bc_upper_bound": 1e2,
                    "predict_epsilon": False,
                    "consistency": False,
                    "debug": True
                }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul04_weight():
    file_paths = []
    job_list = []
    env = ["antmaze-umaze-v0"]
    Ts = [2, 8]
    weights = [0.1, 1.0]
    config_dir = "configs/sample-td3-antmaze/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        # for bc_weight in bc_weights:
        for weight in weights:
            for T in Ts:
                job_id = f"{env_name[:6]}-td3-bcw{weight}-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "discount2": 0.999,
                    "coef": 1.0,
                    "seed": 0,
                    "T": T,
                    "algo": "td3",
                    "env_name": env_name,
                    "bc_weight": weight,
                    "tune_bc_weight": False,
                    "name": job_id,
                    "id": job_id,
                    "bc_lower_bound": 1e-2,
                    "bc_decay": 0.995,
                    "value_threshold": 2.8e-4,
                    "bc_upper_bound": 1e2,
                    "predict_epsilon": False,
                    "consistency": False,
                    "debug": True
                }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul10_bc_reg():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-ddd-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-ddd-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ddd",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul11_dac_reg():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-dac-con-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-dac-con-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])


def jul11_dac_walker_hopper():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2", "walker-medium-v2", "hopper-medium-v2"]
    env = ["hopper-medium-v2", "walker2d-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/sample-dac-con-low/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-dac-con-bcw-low-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.5,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
                "consistency": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul18_bc():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2", "walker-medium-v2", "hopper-medium-v2"]
    # env = ["hopper-medium-v2", "walker2d-medium-v2"]
    env = ["halfcheetah-medium-v2", "halfcheetah-medium-expert-v2"]
    Ts = [1, 2, 4, 8]
    # config_dir = "configs/sample-dac-con-low/"
    config_dir = "configs/bc_compare/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[-16:]}-bc-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "seed": 0,
                "T": T,
                "algo": 'bc',
                "env_name": env_name,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def jul18_bc_sample():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2", "halfcheetah-medium-expert-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/bc_sample_compare/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[-16:]}-bc-sample-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "seed": 0,
                "T": T,
                "algo": 'bc',
                "env_name": env_name,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

def aug19_dac():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/dac-d4rl-online/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"dac-{env_name[:6]}-online-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "update_ema_every": 1, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": True,            
                "need_animation": True, 
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")
    
def aug19_dac_d4rl():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    config_dir = "configs/dac-mujoco-online/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"dac-{env_name[:6]}-mujoco-online-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "update_ema_every": 1, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": True,            
                "need_animation": True, 
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")

def run_multi_py(job, filename, main="main.py", total_gpu=8, directory="inter_result/new_sanity/"):
    i = random.randint(1, total_gpu - 1)
    # if not os.path.exists(directory):
    #     os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, job + ".log")
    command = f"CUDA_VISIBLE_DEVICES={i} nohup python -u {main} --config {filename} id={job} name={job} > {file_path} &"
    # os.system("git add .")
    # os.system(f"git commit -m '{command}''")
    # os.system("git pull origin master")
    # os.system("git push origin master'")
    os.system(command)

def test_multi_py():
    job_list = ["test1", "test2"]
    name_list = ["test1", "test2"]
    age = [20, 21]    
    file_paths = []
    config_dir = "configs/dac-d4rl-online/"
    os.makedirs(config_dir, exist_ok=True)
    for i in range(2):
        job_id = job_list[i]
        file_name = job_id + ".yaml"
        config = {
            'name': name_list[i],
            'age': age[i],
            'id': job_id,
            }
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    print(job_list)
    print(file_paths)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="run_test.py")

def aug20_demo_dac_dql_():
    """
    compare dac and dql
    online setting
    t=1, 2, 4, 8, 16
    git commit code:
    commit 393fef5893939093b95a33fb8b425f38a3c74213 (HEAD -> master)
    Author: lkr <2793158317@qq.com>
    Date:   Sat Aug 19 06:15:29 2023 -0700
    config_dir = "configs/dac-dql-demo-online/"
    """
    file_paths = []
    job_list = []
    # env = ["hopper-medium-v2", "walker2d-medium-v2",]
    # env = ["Demo-v0"]
    env_name = "Demo-v0"
    Ts = [1, 2, 4, 8, 16]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql-demo-online/"
    os.makedirs(config_dir, exist_ok=True)
    # for env_name in env:
    for algo in algos:
        for T in Ts:
            job_id = f"{algo}-demo-online-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "update_ema_every": 1, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": False,
                "need_animation": True, 
                "vis_q": True,
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")
 
def aug20_demo_dac_dql():
    """
    compare dac and dql
    online setting
    t=1, 2, 4, 8, 16
    git commit code:
    commit 393fef5893939093b95a33fb8b425f38a3c74213 (HEAD -> master)
    Author: lkr <2793158317@qq.com>
    Date:   Sat Aug 19 06:15:29 2023 -0700
    config_dir = "configs/dac-dql-demo-online/"
    """
    file_paths = []
    job_list = []
    # env = ["hopper-medium-v2", "walker2d-medium-v2",]
    # env = ["Demo-v0"]
    env_name = "Demo-v0"
    Ts = [1, 2, 4, 8, 16]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql-demo-offline/"
    os.makedirs(config_dir, exist_ok=True)
    for algo in algos:
        for T in Ts:
            job_id = f"{algo}-demo-offline-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "update_ema_every": 1, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": False,
                "need_animation": True, 
                "vis_q": True,
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug22_demo_dac_dql():
    """
    compare dac and dql
    offline setting
    t=1, 2, 4, 8, 16
    git commit code:
    commit 393fef5893939093b95a33fb8b425f38a3c74213 (HEAD -> master)
    Author: lkr <2793158317@qq.com>
    Date:   Sat Aug 19 06:15:29 2023 -0700
    config_dir = "configs/dac-dql-demo-online/"
    """
    file_paths = []
    job_list = []
    # env = ["hopper-medium-v2", "walker2d-medium-v2",]
    # env = ["Demo-v0"]
    env_name = "Demo-v0"
    Ts = [1, 2, 4, 8, 16]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/demo-offline/"
    os.makedirs(config_dir, exist_ok=True)
    for algo in algos:
        for T in Ts:
            job_id = f"{algo}-demo-offline-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "update_ema_every": 1, 
                "online": False,
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": False,
                "need_animation": True, 
                "vis_q": True,
                "num_steps_per_epoch": 1,
                "bc_weight": 1.0,
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug22_demo_dac_dql_scheduler():
    """
    compare different schedulers, linear, vp, cosine
    lack exploration in the online setting 
    online env
    t=4
    git commit code:
    commit d8f5fafe4ecb9bc7016a32d4d20f847810faf808 (HEAD -> master, origin/master, origin/HEAD)
    Author: lkr <2793158317@qq.com>
    Date:   Mon Aug 21 22:33:23 2023 -0700

        offline dac and dql comparison
    config_dir = "configs/dac-dql/demo-offline-scheduler/"
    """
    file_paths = []
    job_list = []
    beta_schedules = ["linear", "vp", "cosine"]
    env_name = "Demo-v0"
    T = 4
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/demo-online-scheduler/"
    os.makedirs(config_dir, exist_ok=True)
    for algo in algos:
        for beta_schedule in beta_schedules:
            job_id = f"{algo}-demo-offline-t{T}"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "beta_schedule": beta_schedule,
                "T": T, 
                "update_ema_every": 1, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb"],
                "env_name": env_name, 
                "d4rl": False,
                "need_animation": True, 
                "vis_q": True,
                }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")

def sanity_check(with_time=False):
    """
    compare different schedulers, linear, vp, cosine
    lack exploration in the online setting 
    online env
    t=4
    git commit code:
    commit d8f5fafe4ecb9bc7016a32d4d20f847810faf808 (HEAD -> master, origin/master, origin/HEAD)
    Author: lkr <2793158317@qq.com>
    Date:   Mon Aug 21 22:33:23 2023 -0700

        offline dac and dql comparison
    config_dir = "configs/dac-dql/demo-offline-scheduler/"
    """
    file_paths = []
    job_list = []
    config_dir = "configs/dac-dql/sanity/"
    for config_file in os.listdir(config_dir):
        # job_id = config_file[:-5] + time.strftime("%H:%M:%S")
        if with_time:
            job_id = config_file[:-5] + time.strftime("%H:%M:%S")
        else:
            job_id = config_file[:-5]
        file_paths.append(os.path.join(config_dir, config_file))
        job_list.append(job_id)
    for ind, job in enumerate(job_list):
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug23_dac_dql_d4rl():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [4, 8]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/d4rl-online/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-nq-online-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")
    
def aug23_dac_dql_d4rl_offline():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [4, 8]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/d4rl-offline/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-discount2-offline-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug25_dac_dql_d4rl_offline():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [4, 8]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/d4rl-offline-q/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-q-offline-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py")

def aug25_dac_dql_d4rl():
    file_paths = []
    job_list = []
    env = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    Ts = [4, 8]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/d4rl-online-q/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-q-online-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug25_dac_dql_d4rl_offline2():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [4, 8]
    algos = ["dac", "dql"]
    config_dir = "configs/dac-dql/d4rl-offline-q2/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-q2-offline-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py")

def bc_sanity_check(with_time=False):
    """
    compare different schedulers, linear, vp, cosine
    lack exploration in the online setting 
    online env
    t=4
    git commit code:
    commit d8f5fafe4ecb9bc7016a32d4d20f847810faf808 (HEAD -> master, origin/master, origin/HEAD)
    Author: lkr <2793158317@qq.com>
    Date:   Mon Aug 21 22:33:23 2023 -0700

        offline dac and dql comparison
    config_dir = "configs/dac-dql/demo-offline-scheduler/"
    """
    file_paths = []
    job_list = []
    config_dir = "configs/dac-dql/bc/"
    for config_file in os.listdir(config_dir):
        job_id = "bc-" + config_file[:-5] + time.strftime("%H:%M:%S")
        file_paths.append(os.path.join(config_dir, config_file))
        job_list.append(job_id)
    for ind, job in enumerate(job_list):
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug26_dac_dql_d4rl_offline():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [4, 8, 16]
    algos = ["dac", "dql", "bc"]
    config_dir = "configs/dac-dql/d4rl-offline-q-ema/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                job_id = f"{algo}-{env_name[:6]}-q-ema-offline-t{T}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_name, 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 0.999,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 0.2,
                    }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")

def aug30_time_computation():
    file_paths = []
    job_list = []
    config_dir = "configs/time-memory/"
    for config_file in os.listdir(config_dir):
        # job_id = "time-" + config_file[:-5] + time.strftime("%H:%M:%S")
        job_id = "time-" + config_file[:-5]
        file_paths.append(os.path.join(config_dir, config_file))
        job_list.append(job_id)
    for ind, job in enumerate(job_list):
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug30_time_computation2():
    file_paths = []
    job_list = []
    config_dir = "configs/time-memory/time5000/"
    for config_file in os.listdir(config_dir):
        # job_id = "time-" + config_file[:-5] + time.strftime("%H:%M:%S")
        job_id = "time5000-" + config_file[:-5]
        file_paths.append(os.path.join(config_dir, config_file))
        job_list.append(job_id)
    for ind, job in enumerate(job_list):
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug30_dac_dql_d4rl_offline():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 4, 8, 32]
    algos = ["dac", "dql"]
    bcws = [1.0, 0.2]
    config_dir = "configs/dac-dql/d4rl-offline-no-bug/"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            for algo in algos:
                for bcw in bcws:
                    job_id = f"{algo}-{env_name[:6]}-t{T}-bcw{bcw}-tvec-offline"
                    file_name = job_id + ".yaml"
                    config = {
                        "algo": algo, 
                        "T": T, 
                        "update_ema_every": 1, 
                        "name": job_id, 
                        "id": job_id, 
                        "predict_epsilon": False, 
                        "format": ['stdout', "wandb", "csv"],
                        "env_name": env_name, 
                        "d4rl": True,            
                        "need_animation": True, 
                        # "discount2": 0.999,
                        "discount2": 1.0,
                        "need_entropy_test": True,
                        "online": False,
                        "num_steps_per_epoch": 1,
                        # "bc_weight": 1.0,
                        "bc_weight": bcw,
                        }
                    job_list.append(
                        job_id)
                    filename = os.path.join(config_dir, file_name)
                    file_paths.append(filename)
                    make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py")

def aug30_check_correct():
    file_paths = []
    job_list = []
    config_dir = "configs/tvec-debug/"
    for config_file in os.listdir(config_dir):
        # job_id = "time-" + config_file[:-5] + time.strftime("%H:%M:%S")
        job_id = "tvec-" + config_file[:-5]
        file_paths.append(os.path.join(config_dir, config_file))
        job_list.append(job_id)
    for ind, job in enumerate(job_list):
        run_multi_py(job, file_paths[ind], main="experiment.py")

def aug31_dql_sanity_check():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    env_d4rls = [("halfcheetah-medium-v2", True), ("Demo-v0", False)]
    # Ts = [1, 4, 16]
    Ts = [1, 4]
    onlines = [True, False]
    # config_dir = f"configs/dql-sanity/change{time.strftime('%H:%M:%S')}/"
    task_id = f"dql-sanity/change{(time.strftime('%H:%M:%S'))}"
    # config_dir = os.path.join("configs", task_id)
    # os.makedirs(config_dir, exist_ok=True)
    # filename = os.path.join(config_dir, "git_log")
    config_dir = f"configs/dql-sanity/"
    # os.system("git log -1 -2 -3 -4 -5 > " + filename)
    for env_d4rl in env_d4rls:
        for T in Ts:
            for online in onlines: 
                job_id = f"dql-{env_d4rl[0][:6]}-t{T}-online{int(online)}-{(time.strftime('%H-%M-%S'))}"
                file_name = job_id + ".yaml"
                if online == False:
                    config = {
                        "algo": "dql", 
                        "T": T, 
                        "update_ema_every": 1, 
                        "name": job_id, 
                        "id": job_id, 
                        "predict_epsilon": False, 
                        "format": ['stdout', "wandb", "csv"],
                        "env_name": env_d4rl[0], 
                        "d4rl": env_d4rl[1],            
                        "need_animation": True, 
                        "discount2": 1.0,
                        "need_entropy_test": True,
                        "online": online,
                        "num_steps_per_epoch": 1,
                        "bc_weight": 1.0,
                        }
                else:
                    config = {
                        "algo": "dql", 
                        "T": T, 
                        "update_ema_every": 1, 
                        "name": job_id, 
                        "id": job_id, 
                        "predict_epsilon": False, 
                        "format": ['stdout', "wandb", "csv"],
                        "env_name": env_d4rl[0], 
                        "d4rl": env_d4rl[1],            
                        "need_animation": True, 
                        "discount2": 1.0,
                        "need_entropy_test": True,
                        "online": online,
                        "bc_weight": 1.0,
                        }
                job_list.append(
                    job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        # run_python_file(job, file_paths[ind], main="experiment.py")
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 -4 -5 > " + git_log)
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def aug31_resample_eval():
    file_paths = []
    job_list = []
    env_d4rls = [("halfcheetah-medium-v2", True), ("Demo-v0", False)]
    Ts = [1, 4]
    algos = ["dql", "dac"]
    # config_dir = f"configs/dql-sanity/change{time.strftime('%H:%M:%S')}/"
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    # config_dir = os.path.join("configs", task_id)
    # os.makedirs(config_dir, exist_ok=True)
    # filename = os.path.join(config_dir, "git_log")
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    # os.system("git log -1 -2 -3 -4 -5 > " + filename)
    for env_d4rl in env_d4rls:
        for T in Ts:
            # for online in onlines: 
            for algo in algos:
                job_id = f"{env_d4rl[0][:6]}-t{T}-algo-{algo}-{(time.strftime('%H-%M-%S'))}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": env_d4rl[0], 
                    "d4rl": env_d4rl[1],            
                    "need_animation": True, 
                    "discount2": 1.0,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    "resample": True,
                    }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 -4 -5 > " + git_log)
        try:
            run_python_file(job, file_paths[ind], main="experiment.py")
        except:
            run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def aug31_resample_eval2():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    # algos = ["dql", "dac"]
    # config_dir = f"configs/dql-sanity/change{time.strftime('%H:%M:%S')}/"
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    # config_dir = os.path.join("configs", task_id)
    # os.makedirs(config_dir, exist_ok=True)
    # filename = os.path.join(config_dir, "git_log")
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    # os.system("git log -1 -2 -3 -4 -5 > " + filename)
    # for env_d4rl in env_d4rls:
    for T in Ts:
        for resample in [True, False]:
            # for online in onlines: 
            # for algo in algos:
                # job_id = f"{env_d4rl[0][:6]}-t{T}-algo-{algo}-{(time.strftime('%H-%M-%S'))}"
                job_id = f"dql-envhalf-t{T}-resample{int(resample)}"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": "halfcheetah-medium-v2", 
                    "d4rl": True,            
                    "need_animation": True, 
                    "discount2": 1.0,
                    "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    "resample": resample,
                    "num_epochs": 10000,
                    "num_steps_per_epoch": 10000,
                    }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 -4 -5 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept3_resample_eval():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    # algos = ["dql", "dac"]
    # config_dir = f"configs/dql-sanity/change{time.strftime('%H:%M:%S')}/"
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    # config_dir = os.path.join("configs", task_id)
    # os.makedirs(config_dir, exist_ok=True)
    # filename = os.path.join(config_dir, "git_log")
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    # os.system("git log -1 -2 -3 -4 -5 > " + filename)
    # for env_d4rl in env_d4rls:
    for T in Ts:
        for resample in [True, False]:
            # for online in onlines: 
            # for algo in algos:
                # job_id = f"{env_d4rl[0][:6]}-t{T}-algo-{algo}-{(time.strftime('%H-%M-%S'))}"
                job_id = f"dql-t{T}-resample{int(resample)}-nopolicydelay-envhalf"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": "halfcheetah-medium-v2", 
                    "d4rl": True,            
                    # "need_animation": True, 
                    "discount2": 1.0,
                    # "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 1,
                    "bc_weight": 1.0,
                    "resample": resample,
                    "num_epochs": 10000,
                    # "num_steps_per_epoch": 10000,
                    }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept3_dql_param():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    # Ts = [4]
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
                job_id = f"dql-t{T}-oldparam"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    # "update_ema_every": 5, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    # "format": ['stdout', "csv"],
                    "env_name": "halfcheetah-medium-v2", 
                    "d4rl": True,            
                    # "need_animation": True, 
                    # "discount2": 1.0,
                    # "need_entropy_test": True,
                    "online": False,
                    "num_steps_per_epoch": 100,
                    "bc_weight": 1.0,
                    "num_epochs": 10000,
                    # "num_steps_per_epoch": 10000,
                }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept3_dql_param_with_v():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
                job_id = f"dql-with-v-t{T}-oldparam"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": "halfcheetah-medium-v2", 
                    "d4rl": True,            
                    "online": False,
                    "num_steps_per_epoch": 5000,
                    "bc_weight": 1.0,
                    "num_epochs": 10000,
                }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept3_dql_wv_test_critic():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"resample-eval/change{(time.strftime('%H:%M:%S'))}"
    config_dir = f"configs/resample-eval/"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
                job_id = f"dql-wv-t{T}-test-critic"
                file_name = job_id + ".yaml"
                config = {
                    "algo": "dql", 
                    "T": T, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": "halfcheetah-medium-v2", 
                    "d4rl": True,            
                    "online": False,
                    "num_steps_per_epoch": 5000,
                    "bc_weight": 1.0,
                    "num_epochs": 10000,
                    "test_critic": True,
                }
                job_list.append(job_id)
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept3_diffusion_bc_unit_test():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"sys_test/sept3_diffusion_bc_unit_test"
    config_dir = f"configs/sys_test/sept3_diffusion_bc_unit_test"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        job_id = f"t{T}-sept3-diffusion-bc-unit-test"
        file_name = job_id + ".yaml"
        config = {
            "algo": "bc", 
            "T": T, 
            "name": job_id, 
            "id": job_id, 
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "env_name": "halfcheetah-medium-v2", 
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "bc_weight": 1.0,
            "num_epochs": 10000,
        }
        job_list.append(job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept3_dql_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"sys_test/sept3_dql_bcw"
    config_dir = f"configs/sys_test/sept3_dql_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        job_id = f"t{T}-sept3-dql-bcw"
        file_name = job_id + ".yaml"
        config = {
            "algo": "dql", 
            "T": T, 
            "name": job_id, 
            "id": job_id, 
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "env_name": "halfcheetah-medium-v2", 
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "bc_weight": 0.2,
            "num_epochs": 10000,
            # "test_critic": True,
        }
        job_list.append(job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept3_dql_dac():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8]
    algos = ["dql", "dac"]
    task_id = f"sys_test/sept3_dql_dac"
    config_dir = f"configs/sys_test/sept3_dql_dac"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
            job_id = f"{algo}-t{T}-sept3-dql-dac"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": 0.2,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept3_dql_dac_online():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8]
    algos = ["dql", "dac"]
    task_id = f"sys_test/sept3_dql_dac_online"
    config_dir = f"configs/sys_test/sept3_dql_dac_online"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
            job_id = f"{algo}-t{T}-sept3-dql-dac-online"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "num_steps_per_epoch": 500,
                "bc_weight": 0.0,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept3_online_demo():
    file_paths = []
    job_list = []
    Ts = [1, 4]
    algos = ["dql", "dac"]
    task_id = f"sys_test/sept3_online_demo"
    config_dir = f"configs/sys_test/sept3_online_demo"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
                job_id = f"{algo}-t{T}-sept3-online-bc"
                file_name = job_id + ".yaml"
                config = {
                    "algo": algo, 
                    "T": T, 
                    # "update_ema_every": 1, 
                    "name": job_id, 
                    "id": job_id, 
                    "predict_epsilon": False, 
                    "format": ['stdout', "wandb", "csv"],
                    "env_name": "Demo-v0", 
                    "d4rl": False,            
                    "need_animation": True, 
                    "vis_q": True,
                    "discount2": 1.0,
                    # "need_entropy_test": True,
                    "online": True,
                    "bc_weight": 0.0,
                    "num_steps_per_epoch": 50,
                    }
                job_list.append(job_id) 
                filename = os.path.join(config_dir, file_name)
                file_paths.append(filename)
                make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)

def sept4_dac():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    # algos = ["dac"]
    bcs = [0.2, 0.5, 1.0]
    task_id = f"sys_test/sept4_dac"
    config_dir = f"configs/sys_test/sept4_dac"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
        # for algo in algos:
            job_id = f"dac-t{T}-bc{bc}-sept4-dac"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept4_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    # algos = ["dac"]
    # bcs = [0.2, 0.5, 1.0]
    task_id = f"sys_test/sept4_dac_bcw"
    config_dir = f"configs/sys_test/sept4_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        # for bc in bcs:
        # for algo in algos:
            job_id = f"dac-t{T}-sept4-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": 0.2,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept5_dql_dac_online():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8]
    algos = ["dql", "dac"]
    task_id = f"sys_test/sept5_dql_dac_online"
    config_dir = f"configs/sys_test/sept5_dql_dac_online"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
            job_id = f"{algo}-t{T}-sept5-dql-dac-online"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "num_steps_per_epoch": 500,
                "bc_weight": 0.0,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept5_dql_dac():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    algos = ["dql", "dac"]
    task_id = f"sys_test/sept5_dql_dac"
    config_dir = f"configs/sys_test/sept5_dql_dac"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
            job_id = f"{algo}-t{T}-sept5-dql-dac"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": 0.2,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
                "batch_size": 512,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept6_dac():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.2, 0.5, 1.0]
    task_id = f"sys_test/sept6_dac"
    config_dir = f"configs/sys_test/sept6_dac"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept6-dac"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept6_dac_online():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8]
    algos = ["dac"]
    task_id = f"sys_test/sept6_dql_dac_online"
    config_dir = f"configs/sys_test/sept6_dql_dac_online"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for algo in algos:
            job_id = f"{algo}-t{T}-sept6-dql-dac-online"
            file_name = job_id + ".yaml"
            config = {
                "algo": algo, 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "num_steps_per_epoch": 500,
                "bc_weight": 0.0,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        # run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept6_dac_debug():
    file_paths = []
    job_list = []
    # Ts = [1, 4, 8, 16]
    Ts = [4]
    # bcs = [0.2, 0.5, 1.0]
    bcs = [1.0]
    task_id = f"sys_test/sept6_dac_debug"
    config_dir = f"configs/sys_test/sept6_dac_debug"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept6-dac-debug"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        # run_python_file(job, file_paths[ind], main="experiment.py")
        run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept6_dac2():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.2]
    task_id = f"sys_test/sept6_dac2"
    config_dir = f"configs/sys_test/sept6_dac2"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept6-dac2"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")


def sept6_dac2():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.2]
    task_id = f"sys_test/sept6_dac2"
    config_dir = f"configs/sys_test/sept6_dac2"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept6-dac2"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept6_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept6_dac_bcw"
    config_dir = f"configs/sys_test/sept6_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept6-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")
        # run_multi_py(job, file_paths[ind], main="experiment.py", directory=dir_path)

def sept7_pre_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept7_pre_dac_bcw"
    config_dir = f"configs/sys_test/sept7_pre_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept7-pre-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "pre-dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
                "ablation": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept7_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept7_dac_bcw"
    config_dir = f"configs/sys_test/sept7_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept7-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "policy_delay": 1,
                "critic_ema": 1,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept7_dac_bcw_coef():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept7_dac_bcw_coef"
    config_dir = f"configs/sys_test/sept7_dac_bcw_coef"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept7-dac-bcw-coef"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "policy_delay": 1,
                "critic_ema": 1,
                "MSBE_coef": 1.0 / (T),
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept8_dac_reg():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    # config_dir = "configs/sample-dac-con-low/"
    # os.makedirs(config_dir, exist_ok=True)
    task_id = f"sys_test/sept8_dac_reg"
    config_dir = f"configs/sys_test/sept8_dac_reg"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept8-dac-reg"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "bc_lower_bound": 1e-2,
                "bc_decay": 0.995,
                "value_threshold": 2.8e-4,
                "bc_upper_bound": 1e2,
                "predict_epsilon": False,
                "consistency": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    # for ind, job in enumerate(job_list):
    #     run_python_file(job, file_paths[ind])
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main.py")

def sept8_pre_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept8_pre_dac_bcw"
    config_dir = f"configs/sys_test/sept8_pre_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept8-pre-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "pre-dac", 
                "pre_eval": True,
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept9_dac_reg():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    # config_dir = "configs/sample-dac-con-low/"
    # os.makedirs(config_dir, exist_ok=True)
    task_id = f"sys_test/sept9_dac_reg"
    config_dir = f"configs/sys_test/sept9_dac_reg"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept9-dac-reg"
            file_name = job_id + ".yaml"
            config = {
                "name": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 100,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    # for ind, job in enumerate(job_list):
    #     run_python_file(job, file_paths[ind])
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main.py")

def sept9_dac_main():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    task_id = f"sys_test/sept9_dac_main"
    config_dir = f"configs/sys_test/sept9_dac_main"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept9-dac-main"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 100,
                "num_epochs": 100000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main.py")

def sept9_dac_run():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 2, 4, 8]
    task_id = f"sys_test/sept9_dac_run"
    config_dir = f"configs/sys_test/sept9_dac_run"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept9-dac-run"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="run.py")

def sept10_dac_pre_main2():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_main2"
    config_dir = f"configs/sys_test/sept10_dac_pre_main2"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-main2"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_pre_dataset():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_dataset"
    config_dir = f"configs/sys_test/sept10_dac_pre_dataset"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-dataset"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_pre_eval():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_eval"
    config_dir = f"configs/sys_test/sept10_dac_pre_eval"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-eval"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "eval_seed": 4096,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_pre_shape():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_shape"
    config_dir = f"configs/sys_test/sept10_dac_pre_shape"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-shape"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_pre_tensor():
    file_paths = []
    job_list = []
    env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_tensor"
    config_dir = f"configs/sys_test/sept10_dac_pre_tensor"
    os.makedirs(config_dir, exist_ok=True)
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-tensor"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_bcw_shape():
    file_paths = []
    job_list = []
    Ts = [1, 8,]
    bcs = [0.01]
    task_id = f"sys_test/sept10_dac_bcw_shape"
    config_dir = f"configs/sys_test/sept10_dac_bcw_shape"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept10-dac-bcw-shape"
            file_name = job_id + ".yaml"
            config = {
                "algo": "dac", 
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "policy_delay": 1,
                "critic_ema": 1,
                "MSBE_coef": 1.0 / (T),
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept10_dac_pre_simp():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_dac_pre_simp"
    config_dir = f"configs/sys_test/sept10_dac_pre_simp"
    os.makedirs(config_dir, exist_ok=True)
    # for env_name in env:
    for pre_dataset in [True, False]:
        for T in Ts:
            job_id = f"pd{int(pre_dataset)}-t{T}-sept10-dac-pre-simp"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": pre_dataset,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_dac_pre_critic():
    file_paths = []
    job_list = []
    Ts = [1, 8]
    env = ["halfcheetah-medium-v2"]
    task_id = f"sys_test/sept10_dac_pre_critic"
    config_dir = f"configs/sys_test/sept10_dac_pre_critic"
    os.makedirs(config_dir, exist_ok=True)
    # for pre_dataset in [True, False]:
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-dac-pre-critic"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "test_critic": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_pre_dac_bcw():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    bcs = [0.01]
    task_id = f"sys_test/sept10_pre_dac_bcw"
    config_dir = f"configs/sys_test/sept10_pre_dac_bcw"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        for bc in bcs:
            job_id = f"dac-t{T}-bc{bc}-sept10-pre-dac-bcw"
            file_name = job_id + ".yaml"
            config = {
                "algo": "pre-dac", 
                "pre_eval": True,
                "T": T, 
                "name": job_id, 
                "id": job_id, 
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "env_name": "halfcheetah-medium-v2", 
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "bc_weight": bc,
                "num_epochs": 10000,
                "test_critic": True,
            }
            job_list.append(job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="experiment.py")

def sept10_pac_pre_simp():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_pac_pre_simp"
    config_dir = f"configs/sys_test/sept10_pac_pre_simp"
    os.makedirs(config_dir, exist_ok=True)
    # for env_name in env:
    for pre_dataset in [True, False]:
        for T in Ts:
            job_id = f"pd{int(pre_dataset)}-t{T}-sept10-pac-pre-simp"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "pac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": pre_dataset,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_pac_pre_critic():
    file_paths = []
    job_list = []
    Ts = [1, 8]
    env = ["halfcheetah-medium-v2"]
    task_id = f"sys_test/sept10_pac_pre_critic"
    config_dir = f"configs/sys_test/sept10_pac_pre_critic"
    os.makedirs(config_dir, exist_ok=True)
    # for pre_dataset in [True, False]:
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept10-pac-pre-critic"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "pac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "test_critic": True,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_pac_pre_mlp():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_pac_pre_mlp"
    config_dir = f"configs/sys_test/sept10_pac_pre_mlp"
    os.makedirs(config_dir, exist_ok=True)
    for pre_dataset in [True]:
        for T in Ts:
            job_id = f"pd{int(pre_dataset)}-t{T}-sept10-pac-pre-mlp"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "pac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": pre_dataset,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_pac_pre_diff():
    file_paths = []
    job_list = []
    # env = ["halfcheetah-medium-v2"]
    Ts = [1, 8]
    task_id = f"sys_test/sept10_pac_pre_diff"
    config_dir = f"configs/sys_test/sept10_pac_pre_diff"
    os.makedirs(config_dir, exist_ok=True)
    for pre_dataset in [True]:
        for T in Ts:
            job_id = f"pd{int(pre_dataset)}-t{T}-sept10-pac-pre-diff"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 0.999,
                "seed": 0,
                "T": T,
                "algo": "pac",
                "env_name": "halfcheetah-medium-v2",
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "pre_dataset": pre_dataset,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="pre_main2.py")

def sept10_nb_t():
    file_paths = []
    job_list = []
    Ts = [1, 8, 16, 64]
    task_id = f"sys_test/sept10_nb_t"
    config_dir = f"configs/sys_test/sept10_nb_t"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        job_id = f"half-t{T}-sept10-nb-t"
        file_name = job_id + ".yaml"
        config = {
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "discount2": 1.0,
            "seed": 0,
            "T": T,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "bc_weight": 0.01,
            "tune_bc_weight": False,
            "name": job_id,
            "id": job_id,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="nb.py")

def sept10_nb_online():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"sys_test/sept10_nb_online"
    config_dir = f"configs/sys_test/sept10_nb_online"
    os.makedirs(config_dir, exist_ok=True)
    for T in Ts:
        job_id = f"half-t{T}-sept10-nb-online"
        file_name = job_id + ".yaml"
        config = {
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "d4rl": True,            
            "online": True,
            "num_steps_per_epoch": 100,
            "discount2": 1.0,
            "seed": 0,
            "T": T,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "tune_bc_weight": False,
            "name": job_id,
            "id": job_id,
            "bc_weight": 0.0,
            "num_epochs": 5000,
            "test_critic": True,
            "ablation": True,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="nb.py")

def sept10_nb_freq():
    file_paths = []
    job_list = []
    # Ts = [1, 8, 16, 64]
    task_id = f"sys_test/sept10_nb_freq"
    config_dir = f"configs/sys_test/sept10_nb_freq"
    os.makedirs(config_dir, exist_ok=True)
    # for T in Ts:
    pfreq = [1, 2, 4]
    for freq in pfreq:
        job_id = f"half-t8-pfreq{freq}-sept10-nb-freq"
        file_name = job_id + ".yaml"
        config = {
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "discount2": 1.0,
            "seed": 0,
            "T": 8,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "bc_weight": 0.01,
            "tune_bc_weight": False,
            "name": job_id,
            "id": job_id,
            "policy_delay": freq,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="nb.py")

def sept11_nb_env():
    file_paths = []
    job_list = []
    Ts = [1, 4, 8, 16]
    task_id = f"sys_test/sept11_nb_env"
    config_dir = f"configs/sys_test/sept11_nb_env"
    os.makedirs(config_dir, exist_ok=True)
    env = ["walker2d-medium-v2", "hopper-medium-v2"]
    for T in Ts:
        for env_name in env:
            job_id = f"{env_name[:6]}-t{T}-sept11-nb-env"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 1.0,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)
        # run_python_file(job, file_paths[ind], main="nb.py")

def sept11_nb_scheduler():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept11_nb_scheduler"
    config_dir = f"configs/sys_test/sept11_nb_scheduler"
    os.makedirs(config_dir, exist_ok=True)
    schedulers = ["origin", "ddim", "dpm_multistep", "ddpm"]
    # if True:
    for scheduler in schedulers:
        job_id = f"half-t8-{scheduler}-sept11-nb-scheduler"
        file_name = job_id + ".yaml"
        config = {
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "n_inf_steps": 4,
            "discount2": 1.0,
            "seed": 0,
            "T": 16,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "bc_weight": 0.01,
            "tune_bc_weight": False,
            "name": job_id,
            "id": job_id,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        # run_python_file(job, file_paths[ind], main="nb.py")
        run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)

def sept12_nb_env():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept12_nb_env"
    config_dir = f"configs/sys_test/sept12_nb_env"
    os.makedirs(config_dir, exist_ok=True)
    Ts = [1, 4, 8, 16]
    for flag in [False, True]:
        for T in Ts:
            job_id = f"hopper-t{T}-flag{int(flag)}-sept12-nb-env"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 1.0,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "dql",
                "env_name": "hopper-medium-v2",
                "bc_weight": 1.0,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
                "consistency": False,
                "ablation": flag,
                "test_critic": flag,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)
        # run_python_file(job, file_paths[ind], main="nb.py")

def sept13_main_env():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept13_main_env"
    config_dir = f"configs/sys_test/sept13_main_env"
    os.makedirs(config_dir, exist_ok=True)
    Ts = [1, 4, 8]
    env = ["walker2d-medium-v2", "hopper-medium-v2"]
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept13-main-env"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 1.0,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "format": ['stdout', "wandb", "csv"],
                "bc_weight": 1.0,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        # run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)
        # run_python_file(job, file_paths[ind], main="nb.py")
        run_python_file(job, file_paths[ind], main="main.py")

def sept13_main_medium_expert():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept13_main_medium_expert"
    config_dir = f"configs/sys_test/sept13_main_medium_expert"
    os.makedirs(config_dir, exist_ok=True)
    Ts = [1, 4, 8]
    env = ["halfcheetah-medium-expert-v2", "walker2d-medium-expert-v2", "hopper-medium-expert-v2"]
    for env_name in env:
        for T in Ts:
            job_id = f"{env_name[:6]}-t{T}-sept13-main-medium-expert"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 1.0,
                "coef": 1.0,
                "seed": 0,
                "T": T,
                "algo": "ql",
                "env_name": env_name,
                "format": ['stdout', "wandb", "csv"],
                "bc_weight": 1.0,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
                "consistency": False,
            }
            job_list.append(
                job_id)
            filename = os.path.join(config_dir, file_name)
            file_paths.append(filename)
            make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_multi_py(job, file_paths[ind], main="main.py", directory=dir_path)
        # run_python_file(job, file_paths[ind], main="nb.py")
        # run_python_file(job, file_paths[ind], main="main.py")

def sept13_nb_scheduler():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept13_nb_scheduler"
    config_dir = f"configs/sys_test/sept13_nb_scheduler"
    os.makedirs(config_dir, exist_ok=True)
    schedulers = ["origin", "ddim", "dpm_multistep", "ddpm"]
    for scheduler in schedulers:
        job_id = f"half-t16-infer4-{scheduler}-sept13-nb-scheduler"
        file_name = job_id + ".yaml"
        config = {
            "predict_epsilon": False, 
            "format": ['stdout', "wandb", "csv"],
            "d4rl": True,            
            "online": False,
            "num_steps_per_epoch": 5000,
            "n_inf_steps": 4,
            "discount2": 1.0,
            "seed": 0,
            "T": 16,
            "algo": "dac",
            "env_name": "halfcheetah-medium-v2",
            "bc_weight": 0.01,
            "tune_bc_weight": False,
            "name": job_id,
            "id": job_id,
            "sampler_type": scheduler,
        }
        job_list.append(
            job_id)
        filename = os.path.join(config_dir, file_name)
        file_paths.append(filename)
        make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        dir_path = os.path.join("inter_result", task_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        git_log = os.path.join(dir_path, "git_log")
        os.system("git log -1 -2 -3 > " + git_log)
        run_python_file(job, file_paths[ind], main="nb.py")
        # run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)

if __name__ == "__main__":
    # jun22_all_env()
    # jun23_discount_all_env()
    # jun23_bc_discount()
    # jun24_bc_weight()
    # jun24_bc_weight_decay()
    # jun25_bc_weight()
    # jun25_consistency()
    # jun26_consistency()
    # jun26_consistency_ql()
    # jun26_vae_ac()
    # jun26_noise_decay()
    # jun26_bc_weight()
    # jun26_bc()
    # jun27_init_noise_decay()
    # jun27_ql_noise()
    # jun27_init_noise_decay_fix()
    # jun28_sota()
    # jun28_consist()
    # jun28_bct()
    # jun28_sota_noise()
    # jun28_sample_bcw()
    # jun28_sample_low_weight()
    # jun28_sota_noise_t()
    # jun28_walker()
    # jun28_hopper()
    # jun29_walker()
    # jun29_hopper()
    # jun30_walker()
    # jun30_hopper()
    # jul03_weight()
    # jul04_weight()
    # jul10_bc_reg()
    # jul11_dac_reg()
    # jul11_dac_walker_hopper()
    # jul11_dac_reg()
    # jul18_bc()
    # jul18_bc_sample()
    # aug19_dac()
    # aug19_dac_d4rl()
    # test_multi_py()
    # aug20_demo_dac_dql()
    # aug22_demo_dac_dql()
    # aug22_demo_dac_dql_scheduler()    
    # sanity_check(with_time=False)
    # aug23_dac_dql_d4rl()
    # aug23_dac_dql_d4rl_offline()
    # sanity_check(with_time=True)
    # aug23_dac_dql_d4rl()
    # aug23_dac_dql_d4rl_offline()
    # aug25_dac_dql_d4rl_offline()
    # aug25_dac_dql_d4rl()
    # aug25_dac_dql_d4rl_offline2()
    # bc_sanity_check()
    # aug26_dac_dql_d4rl_offline()
    # aug30_time_computation()
    # aug30_time_computation2()
    # aug30_dac_dql_d4rl_offline()
    # aug30_check_correct()
    # aug31_dql_sanity_check()
    # aug31_resample_eval()
    # aug31_resample_eval2()
    # sept3_resample_eval()
    # sept3_dql_param()
    # sept3_dql_param_with_v()
    # sept3_dql_wv_test_critic()
    # sept3_diffusion_bc_unit_test()
    # sept3_dql_bcw()
    # sept3_dql_dac()
    # sept3_dql_dac_online()
    # sept3_online_demo()
    # sept4_dac()
    # sept4_dac_bcw()
    # sept5_dql_dac_online()
    # sept5_dql_dac()
    # sept6_dac()
    # sept6_dac_online()
    # sept6_dac_debug()
    # sept6_dac2()
    # sept6_dac_bcw()
    # sept7_pre_dac_bcw()
    # sept7_dac_bcw()
    # sept7_dac_bcw_coef()
    # sept8_dac_reg()
    # sept8_pre_dac_bcw()
    # sept9_dac_reg()
    # sept9_dac_main()
    # sept10_dac_pre_main2()
    # sept10_dac_pre_main2()
    # sept10_dac_pre_dataset()
    # sept10_dac_pre_eval()
    # sept10_dac_pre_shape()
    # sept10_dac_pre_tensor()
    # sept10_dac_bcw_shape()
    # sept10_dac_pre_simp()
    # sept10_dac_pre_critic()
    # sept10_pre_dac_bcw()
    # sept10_pac_pre_simp()
    # sept10_pac_pre_critic() 
    # sept10_pac_pre_diff()
    # sept10_pac_pre_mlp()
    # sept10_nb_t()
    # sept10_nb_online()
    # sept10_nb_freq()
    # sept11_nb_env()
    # sept11_nb_scheduler()
    # sept12_nb_env()
    # sept13_main_env()
    # sept13_main_medium_expert()
    sept13_nb_scheduler()