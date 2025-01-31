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
    # schedulers = ["origin", "ddim", "dpm_multistep", "ddpm"]
    schedulers = ["dpm_multistep"]
    for scheduler in schedulers:
        job_id = f"half-t16-infer4-{scheduler[-4:]}-sept13-nb-scheduler"
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
        # run_python_file(job, file_paths[ind], main="nb.py")
        run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path)

def sept13_main_bc():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept13_main_bc"
    config_dir = f"configs/sys_test/sept13_main_bc"
    os.makedirs(config_dir, exist_ok=True)
    # Ts = [1, 4, 8]
    # env = ["halfcheetah-medium-expert-v2"]
    # for env_name in env:
    for debug in [True, False]:
        # for T in Ts:
            job_id = f"hopper-t8-debug{int(debug)}-sept13-main-bc"
            file_name = job_id + ".yaml"
            config = {
                "discount2": 1.0,
                "coef": 1.0,
                "seed": 0,
                "T": 8,
                "algo": "ql",
                "env_name": "hopper-medium-v2",
                "format": ['stdout', "wandb", "csv"],
                "bc_weight": 1.0,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "predict_epsilon": False,
                "consistency": False,
                "debug": debug,
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
        # run_multi_py(job, file_paths[ind], main="main.py", directory=dir_path)
        run_python_file(job, file_paths[ind], main="main.py")
        
def sept13_nb_vecenv():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept13_nb_vecenv"
    config_dir = f"configs/sys_test/sept13_nb_vecenv"
    os.makedirs(config_dir, exist_ok=True)
    # schedulers = ["dpm_multistep", "origin"]
    schedulers = ["ddpm", "origin"]
    env = ["halfcheetah-medium-v2", "hopper-medium-v2"]
    for scheduler in schedulers:
        for env_name in env:
            job_id = f"{env_name[:4]}-t8-infer8-{scheduler[-4:]}-sept13-nb-vecenv"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": 8,
                "discount2": 1.0,
                "T": 8,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": 0.01,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept14_nb_hopbc():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept14_nb_hopbc"
    config_dir = f"configs/sys_test/sept14_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    for T in [1, 4, 8]:
        for bc_weight in [0.2, 0.4, 0.8, 1.6]:
            infer_steps = min(T, 4) 
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept14-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept14_nb_hopbc_ql():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept14_nb_hopbc"
    config_dir = f"configs/sys_test/sept14_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    # schedulers = ["ddpm"]
    # env = ["hopper-medium-v2"]
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    # for scheduler in schedulers:
        # for env_name in env:
    for T in [1, 4, 8]:
       for bc_weight in [0.1, 0.2, 0.4, 0.8]: 
            infer_steps = min(T, 4) 
            job_id = f"ql-{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept14-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept15_nb_hopbc_ql():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept15_nb_hopbc"
    config_dir = f"configs/sys_test/sept15_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    for T in [1, 4, 8, 16]:
       for bc_weight in [0.2, 0.4, 0.8, 1.6]: 
            infer_steps = min(T, 4) 
            job_id = f"ql-{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept15-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dql",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept15_nb_hopbc():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept15_nb_hopbc"
    config_dir = f"configs/sys_test/sept15_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    for T in [8]:
        for bc_weight in [1.0, 1.2]:
            infer_steps = min(T, 4) 
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept15-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept16_nb_hopbc():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept16_nb_hopbc"
    config_dir = f"configs/sys_test/sept16_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    for T in [8]:
        for bc_weight in [0.01, 0.02, 0.04, 0.05, 0.08, 0.1]:
            infer_steps = min(T, 4) 
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept16-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "bc_weight": bc_weight,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept19_nb_hopbc():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept19_nb_hopbc"
    config_dir = f"configs/sys_test/sept19_nb_hopbc"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    # scheduler = "ddpm"
    scheduler = "origin"
    it = 8
    for T in [8]:
            job_id = f"{env_name[:4]}-t{T}-infer{it}-{scheduler}-dql-sept19-nb-hopbc"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "T": T,
                "n_inf_steps": it,
                "algo": "dql",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "vec_env_eval": True,
                "sampler_type": scheduler,
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

def sept19_nb_mac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept19_nb_mac"
    config_dir = f"configs/sys_test/sept19_nb_mac"
    os.makedirs(config_dir, exist_ok=True)
    # env_name = "hopper-medium-v2"
    env_name = "halfcheetah-medium-v2"
    # scheduler = "ddpm"
    scheduler = "origin"
    infer_steps = 8
    for T in [8]:
        for len_rollout in [1, 4, 8]:
            # infer_steps = min(T, 8) 
            # job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept19-nb-mac"
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept19-nb-mac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
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

def sept19_nb_dac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept19_nb_dac"
    config_dir = f"configs/sys_test/sept19_nb_dac"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    infer_steps = 8
    for T in [8]:
        # for len_rollout in [1, 4, 8]:
            # infer_steps = min(T, 8) 
            # job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept19-nb-dac"
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-sept19-nb-dac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "dac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
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

def sept19_nb_nbmac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept19_nb_nbmac"
    config_dir = f"configs/sys_test/sept19_nb_nbmac"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    # env_name = "halfcheetah-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    infer_steps = 2
    for T in [8]:
        for len_rollout in [1, 2, 4, 8]:
            # infer_steps = min(T, 8) 
            # job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-bc{bc_weight}-sept19-nb-nbmac"
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept19-nb-nbmac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
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

def sept19_nb_hopql():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept19_nb_hopql"
    config_dir = f"configs/sys_test/sept19_nb_hopql"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    it = 2
    for T in [8]:
        for flag in [True, False]:
            job_id = f"{env_name[:4]}-delay2-abla{int(flag)}-t{T}-infer{it}-{scheduler}-dql-sept19-nb-hopql"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "T": T,
                "n_inf_steps": it,
                "algo": "dql",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "vec_env_eval": True,
                "sampler_type": scheduler,
                "test_critic": flag,
                "ablation": flag,
                "policy_delay": 2,
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

def sept20_nb_nbmac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept20_nb_nbmac"
    config_dir = f"configs/sys_test/sept20_nb_nbmac"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    # env_name = "halfcheetah-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    infer_steps = 2
    for T in [8]:
        for len_rollout in [1, 2, 4, 8]:
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept20-nb-nbmac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
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

def sept20_nb_hopql():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept20_nb_hopql"
    config_dir = f"configs/sys_test/sept20_nb_hopql"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    it = 2
    for T in [8]:
        for flag in [True, False]:
            job_id = f"{env_name[:4]}-abla{int(flag)}-t{T}-infer{it}-{scheduler}-dql-sept20-nb-hopql"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "T": T,
                "n_inf_steps": it,
                "algo": "dql",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "vec_env_eval": True,
                "sampler_type": scheduler,
                "test_critic": flag,
                "ablation": flag,
                "policy_delay": 5,
                "bc_weight": 1.0,
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

def sept20_nb_halfmac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept20_nb_halfmac"
    config_dir = f"configs/sys_test/sept20_nb_halfmac"
    os.makedirs(config_dir, exist_ok=True)
    # env_name = "hopper-medium-v2"
    env_name = "halfcheetah-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    infer_steps = 2
    for T in [8]:
        for len_rollout in [1, 2, 4, 8]:
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept20-nb-halfmac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 0.1,
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

def sept20_nb_dql():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept20_nb_dql"
    config_dir = f"configs/sys_test/sept20_nb_dql"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    it = 2
    for T in [8]:
        for flag in [True, False]:
            job_id = f"{env_name[:4]}-abla{int(flag)}-t{T}-infer{it}-{scheduler}-dql-sept20-nb-dql"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "T": T,
                "n_inf_steps": it,
                "algo": "dql",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "vec_env_eval": True,
                "sampler_type": scheduler,
                "test_critic": flag,
                "ablation": flag,
                # "policy_delay": 5,
                "policy_delay": 1,
                "critic_ema": 1,
                "bc_weight": 1.0,
                "update_ema_every": 5,
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

def sept20_nb_resample():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept20_nb_resample"
    config_dir = f"configs/sys_test/sept20_nb_resample"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    scheduler = "ddpm"
    it = 2
    for T in [4, 8]:
        for flag in [True, False]:
            job_id = f"{env_name[:4]}-abla{int(flag)}-t{T}-infer{it}-{scheduler}-resample-sept20-nb-resample"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "discount2": 1.0,
                "T": T,
                "n_inf_steps": it,
                "algo": "dql",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "vec_env_eval": True,
                "sampler_type": scheduler,
                "test_critic": flag,
                "ablation": flag,
                # "policy_delay": 5,
                "policy_delay": 1,
                "critic_ema": 1,
                "bc_weight": 1.0,
                "update_ema_every": 5,
                "resample": True,
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

def sept21_nb_halfmac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept21_nb_halfmac"
    config_dir = f"configs/sys_test/sept21_nb_halfmac"
    os.makedirs(config_dir, exist_ok=True)
    env_name = "hopper-medium-v2"
    # env_name = "halfcheetah-medium-v2"
    scheduler = "ddpm"
    # scheduler = "origin"
    infer_steps = 2
    for T in [8]:
        for len_rollout in [1, 2, 4, 8]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept21-nb-halfmac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 5000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
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

def sept21_nb_pretrain():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept21_nb_pretrain"
    config_dir = f"configs/sys_test/sept21_nb_pretrain"
    os.makedirs(config_dir, exist_ok=True)
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    for env_name in env_names:
        for len_rollout in [1, 2, 4, 8]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept21-nb-pretrain"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 0.02 if env_name == "halfcheetah-medium-v2" else 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
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

def sept21_nb_me():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept21_nb_me"
    config_dir = f"configs/sys_test/sept21_nb_me"
    os.makedirs(config_dir, exist_ok=True)
    # env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    env_names = ["hopper-medium-expert-v2", "halfcheetah-medium-expert-v2", "walker2d-medium-expert-v2"]
    # env_names = ["walker2d-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    for env_name in env_names:
        for len_rollout in [1]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept21-nb-me"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
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

def sept22_nb_tmac():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept22_nb_tmac"
    config_dir = f"configs/sys_test/sept22_nb_tmac"
    os.makedirs(config_dir, exist_ok=True)
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    # env_names = ["halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    for env_name in env_names:
        for len_rollout in [1]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept22-nb-tmac"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 500,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "tmac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                # "vec_env_eval": True,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
                "trajectory": True,
                "state_len": 4,
                "action_len": 3,
                "eval_steps": 2,
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

def sept22_nb_tmacwandb():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept22_nb_tmacwandb"
    config_dir = f"configs/sys_test/sept22_nb_tmacwandb"
    os.makedirs(config_dir, exist_ok=True)
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    for env_name in env_names:
        for len_rollout in [1]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept22-nb-tmacwandb"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "tmac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                # "vec_env_eval": True,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
                "trajectory": True,
                "state_len": 4,
                "action_len": 3,
                "eval_steps": 2,
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

def sept23_diffusion_bc_unit_test():
    file_paths = []
    job_list = []
    # Ts = [8]
    T = 8
    task_id = f"sys_test/sept23_diffusion_bc_unit_test"
    config_dir = f"configs/sys_test/sept23_diffusion_bc_unit_test"
    os.makedirs(config_dir, exist_ok=True)
    # for T in Ts:
    for bc_sequence in [True, False]:
        job_id = f"t{T}-bcs{int(bc_sequence)}-sept23-diffusion-bc-unit-test"
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
            "num_steps_per_epoch": 50000,
            "bc_weight": 1.0,
            "num_epochs": 10000,
            "bc_sequence": bc_sequence,
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
        run_python_file(job, file_paths[ind], main="nb.py")
        # run_multi_py(job, file_paths[ind], main="nb.py", directory=dir_path) 

def sept23_nb_statelen():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept23_nb_statelen"
    config_dir = f"configs/sys_test/sept23_nb_statelen"
    os.makedirs(config_dir, exist_ok=True)
    # env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    len_rollout = 1
    for env_name in env_names:
        for state_len in [2, 8]:
            job_id = f"{env_name[:4]}-t{T}-sl{state_len}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept23-nb-statelen"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "tmac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
                "trajectory": True,
                "state_len": state_len,
                "action_len": 3,
                "eval_steps": 2,
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

def sept23_nb_action_len():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept23_nb_action_len"
    config_dir = f"configs/sys_test/sept23_nb_action_len"
    os.makedirs(config_dir, exist_ok=True)
    # env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    len_rollout = 1
    for env_name in env_names:
        for action_len in [2, 4]:
            job_id = f"{env_name[:4]}-t{T}-al{action_len}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept23-nb-action-len"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "tmac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
                "trajectory": True,
                "state_len": 4,
                "action_len": action_len,
                "eval_steps": 2,
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

def sept23_nb_evalsteps():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept23_nb_evalsteps"
    config_dir = f"configs/sys_test/sept23_nb_evalsteps"
    os.makedirs(config_dir, exist_ok=True)
    # env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    len_rollout = 1
    for env_name in env_names:
        for eval_steps in [1, 3]:
            job_id = f"{env_name[:4]}-t{T}-es{eval_steps}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept23-nb-evalsteps"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "tmac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
                "trajectory": True,
                "state_len": 4,
                "action_len": 3,
                "eval_steps": eval_steps,
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

def sept25_nb_pretrain():
    file_paths = []
    job_list = []
    task_id = f"sys_test/sept25_nb_pretrain"
    config_dir = f"configs/sys_test/sept25_nb_pretrain"
    os.makedirs(config_dir, exist_ok=True)
    env_names = ["halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    for env_name in env_names:
        for len_rollout in [1, 2, 4, 8]:
            job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept25-nb-pretrain"
            file_name = job_id + ".yaml"
            config = {
                "predict_epsilon": False, 
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 50000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": True,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                "resample": True,
                "policy_delay": 1,
                "critic_ema": 1,
                "update_ema_every": 5,
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

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 4000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250, 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}

def run_hyper(env_name, hyper, rollout=1, nautilus=True):
    lr = hyper['lr']
    eta = hyper['eta']
    max_q_backup = hyper['max_q_backup']
    reward_tune = hyper['reward_tune']
    eval_freq = hyper['eval_freq']
    num_steps_per_epoch = eval_freq * 100
    num_epochs = hyper['num_epochs']
    gn = hyper['gn']
    top_k = hyper['top_k']
    # job_id = f"{env_name}-lr{lr}-eta{eta}-maxq{int(max_q_backup)}-{reward_tune[:3]}-eval{eval_freq}-ep{num_epochs}-gn{gn}-top{top_k}-sept26-nb"
    job_id = f"{env_name}-roll{rollout}-28"
    file_name = job_id + ".yaml"
    config = {
        "predict_epsilon": False, 
        "format": ['stdout', "wandb", "csv"],
        "d4rl": True,            
        "online": False,
        "num_steps_per_epoch": num_steps_per_epoch,
        "n_inf_steps": 2,
        "discount2": 1.0,
        "T": 8,
        # "algo": "tmac",
        "algo": "mac",
        "env_name": env_name,
        "tune_bc_weight": False,
        "name": job_id,
        "id": job_id,
        "sampler_type": "ddim",
        # "vec_env_eval": False,
        "vec_env_eval": True,
        "len_rollout": rollout,
        "bc_weight": 1.0,
        "resample": True,
        "policy_delay": 2,
        "critic_ema": 1,
        "update_ema_every": 5,
        # "trajectory": True,
        "batch_size": 1024,
        "trajectory": False,
        "state_len": 4,
        "action_len": 3,
        "eval_steps": 2,
        "lr": lr,
        "eta": eta,
        "max_q_backup": max_q_backup,
        "reward_tune": reward_tune,
        "eval_freq": eval_freq,
        "num_epochs": num_epochs,
        "gn": gn,
        "grad_norm": gn,
        "top_k": top_k,
    }
    filename = os.path.join("configs/sys_test/sept26_pretrain/", file_name)
    make_config_file(filename, config)
    if nautilus:
        run_python_file(job_id, filename, main="nb.py")
    else:
        run_multi_py(job_id, filename, main="nb.py", directory="inter_result/sys_test/sept26_pretrain")

def sept26_nb_ant(hyperparameters: dict):
    antenv = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-large-play-v0",
        "antmaze-large-diverse-v0",
        ]
    replayenv = [
        'halfcheetah-medium-v2',
        'halfcheetah-medium-replay-v2',
        'hopper-medium-replay-v2',
        'walker2d-medium-replay-v2',
    ]
    for env_name in antenv:
        for rollout in [1, 8]:
            run_hyper(env_name, hyperparameters[env_name], rollout=rollout)
    for env_name in replayenv:
        run_hyper(env_name, hyperparameters[env_name])

def sept26_nb_ant_only(hyperparameters: dict):
    antenv = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-large-play-v0",
        "antmaze-large-diverse-v0",
        ]
    for env_name in antenv:
        for rollout in [1, 8]:
            run_hyper(env_name, hyperparameters[env_name], rollout=rollout)

def sept28_nb_ant_only(hyperparameters: dict):
    antenv = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0",
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
        ]
    for env_name in antenv:
        for rollout in [1, 8]:
            run_hyper(env_name, hyperparameters[env_name], rollout=rollout)

def oct19_vp():
    file_paths = []
    job_list = []
    # task_id = f"sys_test/sept23_nb_evalsteps"
    # config_dir = f"configs/sys_test/sept23_nb_evalsteps"
    task_id = f"sys_test/oct19_vp"
    config_dir = f"configs/sys_test/oct19_vp"
    os.makedirs(config_dir, exist_ok=True)
    env_names = ["hopper-medium-v2", "halfcheetah-medium-v2", "walker2d-medium-v2"]
    # env_names = ["hopper-medium-v2", "halfcheetah-medium-v2"]
    scheduler = "ddpm"
    infer_steps = 2
    T = 8
    len_rollout = 1
    for env_name in env_names:
        # for eval_steps in [1, 3]:
            # job_id = f"{env_name[:4]}-t{T}-es{eval_steps}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept23-nb-evalsteps"
            job_id = f"{env_name[:4]}-t{T}-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-oct19-vp"
            file_name = job_id + ".yaml"
            config = {
                "eval_episodes": 20,
                "format": ['stdout', "wandb", "csv"],
                "d4rl": True,            
                "online": False,
                "num_steps_per_epoch": 2000,
                "n_inf_steps": infer_steps,
                "discount2": 1.0,
                "T": T,
                "algo": "mac",
                "env_name": env_name,
                "tune_bc_weight": False,
                "name": job_id,
                "id": job_id,
                "sampler_type": scheduler,
                "vec_env_eval": False,
                "len_rollout": len_rollout,
                "bc_weight": 1.0,
                # "resample": True,
                # "policy_delay": 1,
                "MSBE_coef": 1.0,
                "v_coef": 1.0,
                "q_coef": 1.0,
                # "policy_delay": 2,
                # "critic_ema": 1,
                # "update_ema_every": 5,
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

# def sept25_nb_pretrain():
#     file_paths = []
#     job_list = []
#     task_id = f"sys_test/sept25_nb_pretrain"
#     config_dir = f"configs/sys_test/sept25_nb_pretrain"
#     os.makedirs(config_dir, exist_ok=True)
#     env_names = ["halfcheetah-medium-v2"]
#     scheduler = "ddpm"
#     infer_steps = 2
#     T = 8
#     for env_name in env_names:
#         for len_rollout in [1, 2, 4, 8]:
#             job_id = f"{env_name[:4]}-t{T}-resample-infer{infer_steps}-{scheduler[-4:]}-roll{len_rollout}-sept25-nb-pretrain"
#             file_name = job_id + ".yaml"
#             config = {
#                 "predict_epsilon": False, 
#                 "format": ['stdout', "wandb", "csv"],
#                 "d4rl": True,            
#                 "online": False,
#                 "num_steps_per_epoch": 50000,
#                 "n_inf_steps": infer_steps,
#                 "discount2": 1.0,
#                 "T": T,
#                 "algo": "mac",
#                 "env_name": env_name,
#                 "tune_bc_weight": False,
#                 "name": job_id,
#                 "id": job_id,
#                 "sampler_type": scheduler,
#                 "vec_env_eval": True,
#                 "len_rollout": len_rollout,
#                 "bc_weight": 1.0,
#                 "resample": True,
#                 "policy_delay": 1,
#                 "critic_ema": 1,
#                 "update_ema_every": 5,
#             }
#             job_list.append(
#                 job_id)
#             filename = os.path.join(config_dir, file_name)
#             file_paths.append(filename)
#             make_config_file(filename, config)
#     for ind, job in enumerate(job_list):
#         dir_path = os.path.join("inter_result", task_id)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path, exist_ok=True)
#         git_log = os.path.join(dir_path, "git_log")
#         os.system("git log -1 -2 -3 > " + git_log)
#         run_python_file(job, file_paths[ind], main="nb.py")

def run_hyper_nov16(env_name, hyper, nautilus=True):
    lr = hyper['lr']
    eta = hyper['eta']
    max_q_backup = hyper['max_q_backup']
    reward_tune = hyper['reward_tune']
    eval_freq = hyper['eval_freq']
    num_steps_per_epoch = eval_freq * 100
    num_epochs = hyper['num_epochs']
    gn = hyper['gn']
    top_k = hyper['top_k']
    job_id = f"{env_name}-dql-nov16"
    file_name = job_id + ".yaml"
    T = 8 
    config = {
        "seed": 0,
        "T": T,
        "algo": "ql",
        "env_name": env_name,
        "format": ['stdout', "wandb", "csv"],
        "name": job_id,
        "id": job_id,
        "predict_epsilon": False,
    }
    filename = os.path.join("configs/sys_test/nov16/", file_name)
    make_config_file(filename, config)
    if nautilus:
        # run_python_file(job_id, filename, main="nb.py")
        run_python_file(job_id, filename, main="main.py")
    else:
        # run_multi_py(job_id, filename, main="nb.py", directory="inter_result/sys_test/sept26_pretrain")
        run_multi_py(job_id, filename, main="main.py", directory="inter_result/sys_test/sept26_pretrain")

def nov16_dql():
    envs = [
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-large-diverse-v0",
    ]
    for env in envs:
        run_hyper_nov16(env, hyperparameters[env])

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
    # sept13_nb_scheduler()
    # sept13_main_bc()
    # sept13_nb_vecenv()
    # sept14_nb_hopbc()
    # sept14_nb_hopbc_ql()
    # sept15_nb_hopbc()
    # sept16_nb_hopbc()
    # sept19_nb_hopbc()
    # sept19_nb_mac()
    # sept19_nb_dac()
    # sept19_nb_nbmac()
    # sept19_nb_hopql()
    # sept20_nb_nbmac()
    # sept20_nb_hopql()
    # sept20_nb_halfmac()
    # sept20_nb_dql()
    # sept20_nb_resample()
    # sept21_nb_halfmac()
    # sept21_nb_pretrain()
    # sept21_nb_me()
    # sept22_nb_tmac()
    # sept22_nb_tmacwandb()
    # sept23_diffusion_bc_unit_test()
    # sept23_nb_statelen()
    # sept23_nb_action_len()
    # sept23_nb_evalsteps()
    # sept25_nb_pretrain()
    # sept26_nb_ant(hyperparameters)
    # sept26_nb_ant_only(hyperparameters)
    # sept28_nb_ant_only(hyperparameters)
    # oct19_vp()
    nov16_dql()