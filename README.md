# nano GPT-2
Reproducing GPT-2: on a small peice of internet data.

## Credit: Andrej Karpathy
I followed the lecture by Andrej Karpathy - Let's reproduce GPT-2 (124M):https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10

## Run the model using 8 GPU's: 
torchrun --standalone --nproc_per_node=8 train_gpt2.py 
