import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from .external_knowledge import get_expert_actions, get_kg_set
import pdb



# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACKIANModel(nn.Module, torch_ac.RecurrentACModel):
    """
    The Knowledge-Inclusive Attentive Network.
    """
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, kg_set_name='all', preset_kg_embeds=None):
        super().__init__()
        # Load external knowledge set
        self.env_name = kg_set_name
        self.kg_set = get_kg_set(kg_set_name)
        print("Using the knowledge set: {}".format(kg_set_name))
        print(self.kg_set)

        # Set Dimension
        kg_emb_dim = 8

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Save action space and observation space
        self.obs_space = obs_space
        self.action_space = action_space

        # Define actor's models

        self.actor_base = nn.Linear(self.embedding_size, 64)
        self.actor_Q = nn.Sequential(
            self.actor_base,
            nn.Tanh(),
            nn.Linear(64, kg_emb_dim)
        )
        self.actor_K = nn.Sequential(
            self.actor_base,
            nn.Tanh(),
            nn.Linear(64, kg_emb_dim)
        )
        self.actor_V = nn.Sequential(
            self.actor_base,
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define expert rules (ours)
        self.expert_K = nn.Embedding(len(self.kg_set), kg_emb_dim)

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # Find the expert's actions
        expert_actions = get_expert_actions(obs, self.kg_set, env_name = self.env_name)# (b x # rules x # actions) or (# rules x # actions)

        # The following two implementation variants tackle the entropy imbalance issue, we empirically found that in some cases one is better than another
        if self.action_space.n > 3:
            ## trick: scale the expert_actions to make its entropy learnable
            expert_K_norm = torch.matmul(torch.linalg.norm(self.actor_Q(embedding), dim=1).unsqueeze(1), torch.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(2)# ( b x # rules x 1)
            expert_actions = F.softmax(expert_K_norm * expert_actions, dim=2)

            ## Compute weights
            self.w_i = F.cosine_similarity(self.actor_Q(embedding), self.actor_K(embedding), dim=1).view(-1,1,1)# (b)
            self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(embedding).unsqueeze(1), dim=2).unsqueeze(2)# (b x # rules)
            total_exp_w = torch.exp(self.w_i) + torch.sum(torch.exp(self.W_e), dim=1, keepdim=True)
            self.w_i = torch.exp(self.w_i) / total_exp_w
            self.W_e = torch.exp(self.W_e) / total_exp_w
            inner_pi = F.softmax(self.actor_V(embedding),dim=1)

            x = self.w_i.view(-1,1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=1) #(b x # rules x 1) x (b # rules x # actions)
            
            ## sample dist, when doing pre-softmax
            dist = Categorical(logits=torch.log(x))

        else:
            self.w_i = torch.matmul(self.actor_Q(embedding).unsqueeze(1), self.actor_K(embedding).unsqueeze(2))
            self.W_e = torch.matmul(self.expert_K.weight, self.actor_Q(embedding).unsqueeze(2))
            inner_pi = self.actor_V(embedding)
            x = self.w_i.view(-1,1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=1) #(b x # rules x 1) x (b # rules x # actions)
            
            ## sample dist. when not doing pre-softmax
            dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
