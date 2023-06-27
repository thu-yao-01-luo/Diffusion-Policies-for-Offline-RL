from multiprocessing import Process
import yaml
import os
import random

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


def run_python_file(job, filename):
    command = f"python launch/remote_run.py --job_name {job} main.py --config {filename} --run"
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
        run_python_file(filename)


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
    scales = [1e-1, 1e-2, 1e-3]
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
            # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

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
                # make_config_file(filename, config)
    for ind, job in enumerate(job_list):
        run_python_file(job, file_paths[ind])

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
    jun26_bc()
