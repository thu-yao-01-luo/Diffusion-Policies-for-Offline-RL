import numpy as np
from config import Config

def eval_policy(args:Config, policy, eval_env, algo, eval_episodes=10,):
    # initialize
    d4rl: bool = args.d4rl
    need_animation: bool = args.need_animation
    vis_q: bool = args.vis_q 
    env_name: str = args.env_name
    need_entropy_test: bool = args.need_entropy_test
    scores = []
    lengths = []
    actions_abs = []
    actions = []
    ret = {}
    # test scores and compute normalized scores
    for _ in range(eval_episodes):
        traj_return = 0.
        traj_length = 0
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            actions_abs.append(np.mean(np.abs(action)))
            actions.append(action)
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
            traj_length += 1
        scores.append(traj_return)
        lengths.append(traj_length)
    print("scores", scores)
    print("lengths", lengths)
    avg_reward = np.mean(scores)
    std_reward = np.std(scores)
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    ret["avg_reward"] = avg_reward
    ret["std_reward"] = std_reward
    ret["avg_length"] = avg_length
    ret["std_length"] = std_length
    if len(actions) > 0:
        avg_action = np.mean(actions_abs)
        std_action = np.std(actions_abs)
        ret["avg_action"] = avg_action
        ret["std_action"] = std_action
    if d4rl:
        normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
        avg_norm_score = eval_env.get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)
        ret["avg_norm_score"] = avg_norm_score
        ret["std_norm_score"] = std_norm_score
        utils.print_banner("Normalized score: {:.2f} +/- {:.2f}".format(
            avg_norm_score, std_norm_score))
    # check the local optimality
    if env_name == 'Demo-v0':
        local_opt = True
        state, done = eval_env.reset(), False
        eval_env.set_state(np.array([-5, 0]))
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
        if np.linalg.norm(eval_env.state - np.array([5, 0])) < 2:
            local_opt = False
        ret["local_opt"] = local_opt
    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}")
    # animation 
    if need_animation:
        ims = animation(eval_env, vis_q, policy, algo)
        logger_zhiao.animate(ims, f'{args.algo}_{args.T}_{args.env_name}_bcw{args.bc_weight}.mp4')
    if need_entropy_test:
        # the state is the last state of test(for mujoco, it is suitable)
        if args.env_name == 'Demo-v0':
            state = eval_env.reset()
        elif args.d4rl == True:
            pass
        else:
            raise NotImplementedError
        action_list = []
        for i in range(args.num_entropy_samples):
            # TODO: state is not the last state of test(for mujoco, it is not suitable)
            action = policy.sample_action(np.array(state), noise_scale=args.act_noise) # type:ignore
            action_list.append(action)
        entropy = compute_entropy(action_list)
        ret["entropy"] = entropy
    return ret
