import time

out_dir = 'out-don'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'don'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'don'
init_from = 'gpt2' # this is the GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
