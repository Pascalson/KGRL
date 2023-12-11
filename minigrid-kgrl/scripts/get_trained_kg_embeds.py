import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import os
import utils
from utils import device

import pdb

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--saved_kg", required=True,
                    help="path to save or restore knowledge embeddings (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text, kg_set_name=args.env)#, preset_kg_embeds=args.saved_kg if os.path.isfile(args.saved_kg) else None)
print("Agent loaded\n")

kg_embeds = agent.acmodel.expert_K.weight
kg_strs = agent.acmodel.kg_set
if len(kg_strs) != kg_embeds.size(0):
    kg_strs = agent.acmodel.new_kgs
kg_to_save = {k:kg_embeds[i] for i, k in enumerate(kg_strs)}
try:
    ori_saved_kg = torch.load(args.saved_kg)
    kg_to_save = {**ori_saved_kg, **kg_to_save}
except:
    pass
torch.save(kg_to_save, args.saved_kg)
