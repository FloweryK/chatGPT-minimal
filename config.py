# model
n_vocab = 30000+7
n_seq = 1000
n_layer = 6
n_head = 8
d_emb = 512
d_hidden = 2048
dropout = 0.1
scale = (d_emb // n_head)**(1/2)
assert d_emb % n_head == 0

# dataset
r_split = 0.9

# training
device = 'cuda'
use_amp = True
n_epoch = 50
n_batch = 32
n_accum = 1
lr = 1e-5
warmup_steps = 4000
label_smoothing = 0.1