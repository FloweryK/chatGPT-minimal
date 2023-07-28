# model
n_vocab = 30000+7
n_seq = 1000
n_layer = 6
n_head = 8
d_emb = 512
d_hidden = 2048
dropout = 0.1
scale = (512//8)**(1/2)

# dataset
rate_split = 0.9
target_speaker = '유민상'

# training
device = 'cpu'
n_epoch = 100
n_batch = 8
lr = 0
warmup_steps = 4000
label_smoothing = 0.1

# path
path_vocab = 'src/vocab/kakaotalk_pc_30000.model'
path_data = 'src/dataset/kakaotalk/kakaotalk_pc.txt'
path_weight = 'weights/'