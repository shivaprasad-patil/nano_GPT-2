import os
import sys
import time
import warnings


import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel
import tiktoken

from dataclasses import dataclass
from doctest import master
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from hellaswag import render_example, iterate_examples

import warnings
warnings.filterwarnings('ignore')

os.chdir("<your_project_directory>") # replace with your actual project directory

#torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------

class CausualSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # B=batch size, T=sequence length, C=embedding dimension(n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh = number of heads, hs = head size, and C = number of channels = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64 so nh * hs = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) attention matrix for all the queries and keys)
        
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt((k.size(-1)))) # (B, nh, T, T)
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B, nh, T, T)
        #att = F.softmax(att, dim=-1)
        #y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        return self.c_proj(y)
        return y


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # size of vocabulary 50,000 BPE merges + 256 bytes + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers in the transformer
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension (hidden size)
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # share the token embeddings with the final classifier
        
        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  # scale init for GPT
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T) where B is batch size and T is sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token embeddings and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) + pos_emb # (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # calculate the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from model downloaded from Artifactory"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms do not decay.
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of decayed parameter tensors: {num_decay_params:,} ({len(decay_params)} tensors)")
        print(f"Number of non-decayed parameter tensors: {num_no_decay_params:,} ({len(no_decay_params)} tensors)")
        # Create AdamW optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, 
                                      fused=use_fused)
        return optimizer

# ----------------------------------------------------------------------------------------------------


def load_tokens(filename):
    npt = np.load(filename)  # load the numpy array of tokens
    ptt = torch.tensor(npt, dtype=torch.long)  # convert to PyTorch tensor
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # get the shard filenames
        data_root = "data_tokens"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)  # sort the shards to ensure consistent order
        shards = [os.path.join(data_root, s) for s in shards]  # full paths to the shards
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()  # reset the dataloader to initialize the loading
        
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0  # current shard index
        self.tokens = load_tokens(self.shards[self.current_shard])  # load the first shard
        self.current_position = self.B * self.T * self.process_rank  # current position in the tokens
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]  # get the next B*T+1 tokens
        x = buf[:-1].view(B, T)  # reshape to (B, T) inputs
        y = buf[1:].view(B, T)  # reshape to (B, T) targets
        # advance the position in the tensor
    
        self.current_position += B * T * self.num_processes  # advance the position by B * T * num_processes
        # if the loading the next batch would be out of bounds, reset the position to 0
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)  # move to the next shard
            self.tokens = load_tokens(self.shards[self.current_shard])  # load the next shard
            self.current_position = self.B * self.T * self.process_rank
        return x, y
# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ----------------------------------------------------------------------------------------------------
# attempt to autodetect the device

# run the training loop
# setup DDP (distributed data parallel) if available
# torchrun command sets the environment variable RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a distributed run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    init_process_group(backend='nccl')  # initialize the process group for distributed training
    ddp_rank = int(os.environ['RANK'])  # get the rank of the process
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # get the local rank of the process
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # get the total number of processes
    device = f'cuda:{ddp_local_rank}'  # set the device to the local rank
    torch.cuda.set_device(device)  # set the CUDA device
    master_process = ddp_rank == 0  # this process will do loggging, checkpointing, etc.
else:
    # if not distributed, non-ddp run vanilla
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

torch.manual_seed(1337) # set the seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)  # if using GPU
    
total_batch_size = 524288 # 2^19 tokens in the dataset, 2^19 / 1024 = 512 batches of size 1024 ~ 0.5M tokens
B = 8  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)  # number of gradient accumulation steps
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print(" I am GPU ", ddp_rank)

# Open AI runs with B = 16, T = 1024
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')  # set the float32 matmul precision to high for better performance

# create model
model = GPT(GPTConfig(vocab_size=50304))  # initialize a model from scratch so random weights
model.to(device)
use_compile = True #False # torch.compile interferes with HellaSwag eval & Generation, so we disable it for now
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])  # wrap the model in DDP if using distributed training
raw_model = model.module if ddp else model  # get the raw model if using DDP, otherwise just the model
    
max_lr = 6e-4  # set the maximum learning rate
min_lr = max_lr * 0.1  # set the minimum learning rate
warmup_steps = 715 # number of warmup steps for the learning rate scheduler
max_steps = 19073  # total number of training steps
def get_lr(it):
    # 1) linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iter, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # cosine decay
    # return the learning rate scaled by the cosine decay coefficient
    return min_lr + coeff * (max_lr - min_lr)  # scale the learning rate down to min_lr
    
# optimize!
optimizer =  raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create the log Directory we will write checkpoints and logs to
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # create the log directory if it doesn't exist
log_file = os.path.join(log_dir, f"log.txt")  # log file for this rank
with open(log_file, 'w') as f: # open the log file for writing
    pass

for step in range(max_steps):
    t0 = time.time()  # start the timer
    last_step = (step == max_steps - 1)  # is this the last step?
    
    # once in a while, evaluate the model on the validation set
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()  # reset the validation loader to the start
        with torch.no_grad():
            val_loss_accum = 0.0  # initialize the validation loss accumulator
            val_loss_steps = 20 # number of steps to average the validation loss over
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)  # move to GPU if available
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps  # scale the loss by the number of gradient accumulation steps
                val_loss_accum += loss.detach()  # accumulate the validation loss
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                # print the validation loss every 250 steps
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:  # append to the log file
                    f.write(f"step {step} | validation loss: {val_loss_accum.item():.4f}\n")
                
                # save a checkpoint every 5000 steps
                """if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),  # save the model state dict
                        'config': raw_model.config,  # save the model config
                        'step': step,  # save the current step
                        'val_loss': val_loss_accum.item(),  # save the validation loss
                    }
                    torch.save(checkpoint, checkpoint_path)"""

        # once in a while evaluate hellaswag
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_correct_norm = 0  # number of correct predictions normalized by the number of samples
            num_total = 0 # total number of samples
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and label
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)  # move to GPU if available
                mask = mask.to(device)  # move to GPU if available
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits = model(tokens).logits
                    pred_norm = get_most_likely_row(tokens, mask, logits)  # get the most likely row
                num_total += 1  # increment the total number of samples
                num_correct_norm += int(pred_norm == label)  # increment the number of correct predictions
                # reduce the stats across all processes
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)  # sum
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)  # sum
                    num_total = num_total.item()  # convert to Python int
                    num_correct_norm = num_correct_norm.item()  # convert to Python int
                acc_norm = num_correct_norm / num_total # calculate the accuracy
                if master_process:
                    # print the accuracy every 250 steps
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    with open(log_file, 'a') as f:  # append to the log file
                        f.write(f"step {step} | hellaswag acc_norm: {acc_norm:.4f}\n")
                
    # once in a while, generate some text to see how the model is doing (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4  # number of sequences to generate
        max_length = 32  # maximum length of the generated sequences
        tokens = enc.encode("Hello, I'm a language model,") # encode the prefix
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) 
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)  # create a random number generator for sampling
        sample_rng.manual_seed(42 + ddp_rank)  # set the seed for reproducibility
        while xgen.size(1) < max_length:
            # forward the model to get the logits for the next token
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(xgen) # (xgen)[0]for pre-trained model
                logits = logits[:, -1, :] # (B, vocab_size), only take the last token's logits
                probs = F.softmax(logits, dim=-1) # (B, vocab_size) probabilities of the next token
                # do top-k sampling of 50 (huggingface default)
                # topk_probs here becomes (5, 50) and topk_indices becomes (5, 50)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices from the top-k indices
                xcol = torch.gather(topk_indices, dim=-1, index=ix) # (B, 1)
                # append the selected token to the input sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    # training loop
    model.train()  # set the model to training mode
    optimizer.zero_grad()  # zero the gradients
    loss_accum = 0.0  # initialize the loss accumulator
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()  # get the next batch of data
        x, y = x.to(device), y.to(device)  # move to GPU if available
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)  # forward the model with input x and target y
        loss = loss / grad_accum_steps  # scale the loss by the number of gradient accumulation steps
        loss_accum += loss.detach()  # accumulate the loss
        if ddp:
            # sync gradients only on the last micro step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) 
        loss.backward()  # backpropagate the loss
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the loss across all processes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the gradients to avoid exploding gradients
    # determine the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # set the learning rate for the optimizer
    optimizer.step()  # update the model parameters
    torch.cuda.synchronize()  # synchronize the GPU if using CUDA
    t1 = time.time()  # end the timer
    dt = (t1 - t0)  # calculate the time taken for this iteration in milliseconds
    tokens_processed = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)  # calculate the number of tokens processed in this iteration
    tokens_per_sec = tokens_processed / dt  # calculate the number of tokens processed per second
    if master_process:
        # print the loss and time every iteration
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} |lr {lr:.4e} | norm: {norm:.4f}| dt: {dt * 1000:.2f}ms, tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, 'a') as f:  # append to the log file
            f.write(f"step {step} | training loss: {loss_accum.item():.6f}\n")
            
# destroy the process group if using distributed training 
if ddp: 
    destroy_process_group()  

# Script to run the model.   
# torchrun --standalone --nproc_per_node=8 train_gpt2_fineweb_edu.py   # 8 is the number of GPUs