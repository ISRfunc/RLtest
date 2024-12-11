import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__()

class Actor(nn.Module):
    
    def __init__(self, arch: nn.Module, n_joints: int) -> None:
        super(Actor, self).__init__()
 
        self.encoder = arch 
        self.height = self.encoder.height 
        self.width = self.encoder.width
        self.n_joints = n_joints

        self.hidden_dim = self.encoder.hidden_dim_mlp + 8

        self.mean_head = nn.Linear(in_features=self.hidden_dim, out_features=self.n_joints)
        self.std_head  = nn.Linear(in_features=self.hidden_dim, out_features=self.n_joints)
        self.softplus = nn.Softplus()


    def forward(self, state: torch.Tensor):
        #features = torch.flatten(self.encoder(state), start_dim=1)

        features = torch.flatten(self.encoder(state["img"]), start_dim=1)

        extras = state["joints"]

        features = torch.concatenate( (features, extras) , 1)

        action_means = self.mean_head(features)
        action_stds  = self.softplus(self.std_head(features))
 
        return action_means, action_stds

    def _count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        
class Critic(nn.Module):

    def __init__(self, arch: nn.Module):
        super(Critic, self).__init__()

        self.encoder = arch 
        self.height = self.encoder.height 
        self.width = self.encoder.width

        self.hidden_dim = self.encoder.hidden_dim_mlp + 8

        self.head = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, state): #: torch.Tensor
        #features = torch.flatten(self.encoder(state), start_dim=1)

        features = torch.flatten(self.encoder(state["img"]), start_dim=1)

        extras = state["joints"]

        features = torch.concatenate( (features, extras) , 1)

        value  = self.head(features)

        return value
    
    def count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PPO():
    def __init__(self, actor: nn.Module, critic: nn.Module, clip: float=0.17, lr: float=3e-4, values_loss_coeff: float=0.5, entropy_loss_coeff: float=0.002, *, logging: torch.utils.tensorboard.writer.SummaryWriter) -> None: #clip: float=0.2
        
        # initializing ppo parameters 
        self.actor = actor
        self.critic = critic 
        self.clip = clip
        self.value_clip_param = 0.2
        

        # loss coefficient 
        self.values_loss_coeff = values_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        
        # initializing optimizer 
        self.lr = lr
        self.optimizer = torch.optim.Adam(params= list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

        # logging 

        self.writer = logging  # TensorBoard writer

        self.step = 0  # global step counter for logging
        
        self.epsCount = 0

    def update(self, rollout, mini_batch_size): 

        #states: torch.Tensor, actions: torch.Tensor, log_probs_old, advantages, returns
        num_steps, num_processes = rollout.rewards.size()[0:2]
        batch_size = num_processes * num_steps       
        num_mini_batch = batch_size // mini_batch_size
        print("num_mini_batch is: ", num_mini_batch)

        self.epoch = num_mini_batch



        advantages = rollout.returns - rollout.value_preds
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        data_generator = rollout.batchSample(
                        advantages, mini_batch_size)

        
        #for i in range (self.epoch): 
        for sample in data_generator:

            value_preds = sample['value_preds'].to("cuda")
            returns = sample['returns'].to("cuda")
            advantages = sample['adv_targ'].to("cuda")

            actions = sample['actions'].to("cuda")
            log_probs_old = sample['old_action_log_probs'].to("cuda")

            states = sample['obs']
            for key in states:
                states[key] = states[key].to("cuda")

            """"
            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], sample['actions'],
                        extras=sample['extras']
                    )
            """


            means, stds = self.actor(states)

            dists = torch.distributions.Normal(loc=means, scale=stds)
            log_probs_new = dists.log_prob(actions).sum(dim=1)
            entropy = dists.entropy().mean()
            
            values = self.critic(states)
            
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio*advantages
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip)*advantages

            # defining losses 
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = nn.MSELoss()(values, returns)
            entropy_loss = -entropy

            #################################################################
            value_pred_clipped = value_preds + \
                                (values - value_preds).clamp(
                                    -self.value_clip_param, self.value_clip_param)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped
                                    - returns).pow(2)
            value_loss = .5 * torch.max(value_losses,
                                        value_losses_clipped).mean()
            #################################################################

            loss = policy_loss + self.values_loss_coeff*value_loss + self.entropy_loss_coeff*entropy_loss

            # backprop 
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()

            # logging
            self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.step)
            self.writer.add_scalar('Loss/Value', value_loss.item(), self.step)
            self.writer.add_scalar('Loss/Entropy', entropy_loss.item(), self.step)
            self.writer.add_scalar('Values/Mean', values.mean().item(), self.step)
            self.writer.add_scalar('Values/Std', values.std().item(), self.step)
            
            self.step += 1

        self.epsCount += 1
        self.writer.add_scalar('Return', rollout.returns[0].mean().item(), self.epsCount)

    def sample_action(self, state: torch.Tensor):
        """Sample action given state"""
        means, stds = self.actor(state)

        dists = torch.distributions.Normal(loc=means, scale=stds)

        action = dists.sample()

        return action 

    def save(self, path):
        """Save the model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load the model parameters."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Debug
if __name__=="__main__":


    pass




