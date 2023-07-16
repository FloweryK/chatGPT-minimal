# model
n_vocab = 8000+7
n_seq = 1000
n_layer = 6
n_head = 8
d_emb = 512
d_hidden = 2048
dropout = 0.1
scale = (512//8)**(1/2)

# training
n_epoch = 30
n_batch = 16
device = 'cuda'
lr = 0