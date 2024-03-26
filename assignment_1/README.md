# DI 725: Transformers and Attention-Based Deep Networks

## An Assignment for Implementing Transformers in PyTorch

The purpose of this notebook is to guide you through the usage of sample code.

This notebook follows the baseline prepared by Andrej Karpathy, with a custom dataset (Don-Quixote by Cervantes). This version of the code, called [nanoGPT](https://github.com/karpathy/nanoGPT), is a revisit to his famous [minGPT](https://github.com/karpathy/minGPT).

Structure of the starter code:
config: The configuration files for various applications, controlling how nanoGPT behaves.
data: Use prepare.py files there to prepare the data (download and pre-process)
prompt: Folder to store prompts or inputs to be fed to a model
bench.py: A much shorter version of train.py for benchmarking
configurator.py: A support script for configurations
model.py: THE main script where transformer architecture is defined altogether!
sample.py: Used for inference or sampling using a previously trained model
train.py: The script used for training models
Starter_Code.ipynb: The starter code demonstrating how to use this repository.


### Author:
* Ümit Mert Çağlar

## Requirements
Install requirements for your environment, comment out for later uses.

Dependencies:

- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
-  `transformers` for huggingface transformers (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets (to download + preprocess datasets)
-  `tiktoken` for OpenAI's fast BPE code
-  `wandb` for optional logging
-  `tqdm` for progress bars

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Quick Start

The fastest way to get started to transformers, apart from following the labs of DI725, is to use a small model and dataset. For this purpose, we will start with training a character-level GPT on the Don-Quixote by Cervantes. The code will download a single file (2MB) and apply some transformations. Examine the code [prepare.py](data/don_char/prepare.py).

Use the following to prepare the don-quixote novel treated in character level:

```
python data/don_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train our own GPT. The size of the GPT model depends on the computational resources. It is advised to have a GPU for heavy works, and to train lightweight and evaluate and infer models with a CPU.

Small scale GPT with the settings provided in the [config/train_don_char.py](config/train_don_char.py) config file will be trained with the following code:

```
python train.py config/train_don_char.py
```

We are training a small scaled GPT with a context size of up to 256 characters, 384 feature channels, 6 layers of transformer with 6 attention heads. On one GTX 3070 GPU this training run takes about 10 minutes and the best validation loss is 1.1620. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-don-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:
```
python sample.py --out_dir=out-don-char
```
This generates a few samples, for example:

```
“I grant all that,” said the governor; “it’s not in a low voice

but not yet forget that there’s none of it the poor in the world; I’ll

like to take special to have been no one to write out the stone of

patience to the village.”

```

It is pretty nice to have a GPT in a few minutes of character level training! Better results can be achieved possibly by hyperparameter tuning and finetuning (transfer learning) from a pre-trained model.

## Quick start with less resources

If we are [low on resources](https://www.youtube.com/watch?v=rcXzn6xXdIc), we can use a simpler version of the training, first we need to set compile to false, this is also a must for Windows OS for now. We also set the device to CPU. The model that is trained in 10 minutes for a starter grade GPU, will be trained in a much longer time, so we can also decrease the dimensions of our model as follows:

```
python train.py config/train_don_char.py --device=cpu --out_dir="out-don-small-char" --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~5 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```
python sample.py --out_dir=out-don-small-char --device=cpu
```
Generates samples like this:

```
Sancho nother with this then of everantan has for five he enver any

shal were than as in though they and I knight the sther his a jlage,

and mad priled and squiel a hist to in feet she took and and sersse to her of

Marest and good was pefor rubt some by than lave from his dintat all

pack that he remants to goost ever to him arestiance of it the were to who

which mom, worly gane for he sporen gort he was roosion, and be that

it thou, so so he kniders what the and him of him dest us on shart
```

*Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.*

*Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.*

## Finetuning

Finetuning or transfer learning is a precious method of achieving better models thanks to pre-trained models. Finetuning GPT models is just as simple as training from scratch! We will now download the Don-Quixote (again) but this time we will define it with tokens (using OpenAI's BPE tokenizer) instead of characters.

Run an example finetuning like:

```
python train.py config/finetune_don.py
```

This will load the config parameter overrides in `config/finetune_don.py`. Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. Model architecture is changable to `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) and can be decreased in size by the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-don` by default, per the config file. You can then run the code in `sample.py --out_dir=out-don`:
```
* * All creatures that enter the world below may so far as want to observe the rules of their own land, and may obey them under the hand of their lord, and may not follow others below.

* * *

THE PORT COLLIDATES,

- * *

ON the light, and the light to the dark, and the darkness to the light, and the darkness to the darkness, were the present-day laws of monarchy, whose lordship they approved in their faces and hearts. From this moment on, however, they had no other representation to give than that of their master, who, for all that was said or heard, had reached the height of his power.

The king's hand, though at times little more than a finger of his, required no more than a finger of his, and that power was, that of holding his eye, and the other of his, in his own, body.

When this was spoken of, it was a simple and noble quibble, and the subject of this was so as to admit of the few who had any forsemination, and the few who had the most to go on.

The time did not come for a thought of this, and for a moment the very thought of it seemed to fall to the ground.

But that thought did not come to pass; though the king was not speaking of the king, it came to pass that the king, with all his might, and all his cunning, and no other sense, and without any understanding, and without any desire for the utmost of his services, and without any desire to put an end to his own glory, and without any desire to hide his triumph, had found the time to say that this was what he thought on the subject of religion; that it was what he thought, and according as it seemed to him to be as good or better to him than to the other kings, and he was in no sense a king, for it seemed to him he could never have any more power than he had to be; that it was a matter of his will and power; and that it was all a matter of his will, for he was determined that this look and that to which he might have been given to hold it was the best in himself.

And so it was that the king, who was all around him, and all around him; and so
```

# Inference and Sampling
Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```
python sample.py --out_dir=out-don --start="Explain the relationship between Don Quixote and Sancho Panza" --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g.:

```
python sample.py --start=FILE:prompt.txt

```
I hope you will enjoy with the GPT as much as I did!

## Efficiency notes

*For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.*

*Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!*


## Troubleshooting

*Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.*

*For some context on this repository, GPT, and language modeling it might be helpful to watch [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.*

## Acknowledgements

This code is a fork from Andrej Karpathy's introductory [NanoGPT repository](https://github.com/karpathy/nanoGPT), which is an updated form of minGPT.

# Further Experiments

(Completely Optional)

For further experiments, you can reproduce the GPT-2, which is fairly old yet compared to the newer versions while still powerful.


## Reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `$ python sample.py`.

Finally, to train on a single GPU simply run the `$ python train.py` script. Have a[readme.md](data%2Fdon%2Freadme.md) look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## Baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```
python train.py eval_gpt2
python train.py eval_gpt2_medium
python train.py eval_gpt2_large
python train.py eval_gpt2_xl
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.
