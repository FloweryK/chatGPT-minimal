# model
n_vocab = 8000+7
n_seq = 1000
n_layer = 6
n_head = 8
d_emb = 512
d_hidden = 2048
dropout = 0.1
scale = (512//8)**(1/2)

# dataset
rate_split = 0.9

# training
n_epoch = 100
n_batch = 8
device = 'cuda'
lr = 0
warmup_steps = 4000
label_smoothing = 0.1