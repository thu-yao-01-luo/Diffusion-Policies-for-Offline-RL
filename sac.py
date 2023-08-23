import os
from torch.optim import Adam
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        # noise = torch.randn([state.shape[0], self.num_actions]).to(device=state.device)
        # x = F.relu(self.linear1(torch.concat([state, noise], dim=1)))
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        noise = noise.to(device=mean.device)
        action = mean + noise
        return action, torch.tensor(0.), mean

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_range = [action_space.low, action_space.high]

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.value = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.value_target = ValueNetwork(num_inputs, args.hidden_size).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=args.lr)
        hard_update(self.value_target, self.value)
        self.determine = args.determine
        if args.determine == False:
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        else:
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            # action, _, _, _ = self.policy.sample(state)
            action = self.policy.sample(state)[0]
        else:
            # _, _, action, _ = self.policy.sample(state)
            # action = torch.tanh(action)
            action = self.policy.sample(state)[2]
        action = action.detach().cpu().numpy()[0]
        return self.rescale_action(action)

    def sample_action(self, state, noise_scale=0.0):
        if noise_scale == 0.0:
            return self.select_action(state, eval=True)
        else:
            return self.select_action(state, eval=False)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
                (self.action_range[1] + self.action_range[0]) / 2.0

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # o, a, o2, r, d
        # state, action, next_state, reward, not_done
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            vf_next_target = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.determine == False:
            # assert self.policy is GaussianPolicy, "not guassian"
            pi, log_pi, mean, log_std = self.policy.sample(state_batch) # type: ignore
        else:
            pi, log_pi, mean, = self.policy.sample(state_batch)[:3]

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # Regularization Loss
        if self.determine == False:
            reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean()) # type: ignore
        else:
            reg_loss = 0.001 * (mean.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        vf = self.value(state_batch)
        
        with torch.no_grad():
            vf_target = min_qf_pi - (self.alpha * log_pi)
            # vf_target = min_qf_pi.detach() 

        vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        self.value_optim.zero_grad()
        vf_loss.backward()
        self.value_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)

        return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    def train(self, iterations, replay_buffer, batch_size, log_writer, t): 
        metric = {"vf_loss": [], "qf1_loss": [], "qf2_loss": [], "policy_loss": []}
        for iter in range(iterations):
            # vf_loss, qf1_loss, qf2_loss, policy_loss = self.update_parameters(replay_buffer, batch_size, t)
            vf_loss, qf1_loss, qf2_loss, policy_loss = self.update_parameters(replay_buffer, batch_size, iter)
            """ Log """
            metric['vf_loss'].append(vf_loss)
            metric['qf1_loss'].append(qf1_loss)
            metric['qf2_loss'].append(qf2_loss)
            metric['policy_loss'].append(policy_loss)
            if log_writer is not None:
                log_writer.add_scalar('vf_loss', vf_loss, t)
                log_writer.add_scalar('qf1_loss', qf1_loss, t)
                log_writer.add_scalar('qf2_loss', qf2_loss, t)
                log_writer.add_scalar('policy_loss', policy_loss, t)
        return metric
           
    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.value.state_dict(), value_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            # TODO: fix this
            self.value.load_state.dict(torch.load(value_path)) # type: ignore
