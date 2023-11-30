# transformer-chatbot

> :bulb: A simple chatbot using transformer

This project provides:

- **_One_** `dataset` class:

  - which uses `sentencepiece` as its tokenizer, and requires the dataset as a csv file (formats are described below)

- **_Two_** `Trasnformer` models:

  - an usecase of `nn.Transformer`, and a built-from-scratch transformer with explanations.

- **_Three_** optimization settings for training:
  - [Automatic Mixed Precision to FP16](https://pytorch.org/docs/stable/notes/amp_examples.html)
  - [Gradient Accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation)
  - Warmup scheduler

<br/>

<br/>

## How to install

```bash
# (optional: create a virtualenv)
>> virtualenv venv -p python3
>> source venv/bin/activate

# install with requirements.txt
>> pip install -r requirements.txt
```

<br/>

<br/>

## How to use

#### 1. Prepare your dataset as a csv file

- Prepare your dataset as a `csv` file with headers as 'Q' and 'A'. It might look like this:

  | Q                                                                              | A                                                 |
  | ------------------------------------------------------------------------------ | ------------------------------------------------- |
  | Hey, you said one out of fourteen million, we'd win, yeah? Tell me this is it. | If I tell you what happens, it won't happen.      |
  | Did she have any family?                                                       | Yeah. Us.                                         |
  | Don't do anything stupid until I come back.                                    | How can I? You're taking all the stupid with you. |
  | ...                                                                            | ...                                               |

<br/>

<br/>

#### 2. Train the chatbot

```bash
>> python train.py -d {path-to-the-csv}
```

| flag       | description         | example                         | default  |
| ---------- | ------------------- | ------------------------------- | -------- |
| -d, --data | path to the dataset | -d src/dataset/chatbot_data.csv | Required |

- this will create three files under `runs/{datetime}_vocab={vocab_size}...`
  1. a vocab file used in the sentencepiece (which might be needed when inferencing)
  2. a model weight file (which might also be needed when inferencing)
  3. a tensorboard event file

<br/>

<br/>

#### 3. (Optional) Run tensorboard and monitor metircs

```bash
>> tensorboard --logdir=runs/
```

<br/>

<br/>

#### 4. Talk to the trained model

```bash
>> python chatbot.py -v {path-to-vocab} -w {path-to-weight}
```

| flag         | description                           | example                           | default  |
| ------------ | ------------------------------------- | --------------------------------- | -------- |
| -v, --vocab  | path to the sentencepiece vocab model | -v runs/20230830\_.../vocab.model | Required |
| -w, --weight | path to the trained weight            | -w runs/20230830\_.../model_49.pt | Required |

```bash
>> python chatbot.py -v {path-to-vocab} -w {path-to-weight}
text: who are you?

[2, 187, 124, 16, 19991, 3] # encoded pieces of the input text
who are you?	# input text

[2, 8, 19984, 19981, 20, 7805, 19976, 3] # encoded pieces of the answer
i'm the creator.	# answer
```

<br/>

<br/>

## Performance Test Results

- config used for testing

  ```python
  # model
  n_vocab = 20000+7
  n_seq = 1000
  n_layer = 6
  n_head = 8
  d_emb = 512		# also known as d_model
  d_hidden = 2048
  dropout = 0.1

  # dataset
  r_split = 0.9

  # training
  device = 'cuda'
  use_amp = True
  n_epoch = 50
  lr = 1e-5			# not so meaningful when using a warmup scheduler
  warmup_steps = 4000
  label_smoothing = 0.1
  ```

- results (using only one gpu - GTX 1070 8GB)

  | Dataset                            | dataset size | `n_batch` / `n_accum` | BLEU (train / test) | Time    | GPU Memory |
  | ---------------------------------- | ------------ | --------------------- | ------------------- | ------- | ---------- |
  | Cornell movie-dialogs corpus       | 221,616      | 8 / 32                | 12.7 / 2.98         | 25.7 hr | 5.89 GB    |
  | Korean Chatbot Data                | 11,823       | 32 / 1                | 96.3 / 33.8         | 0.4 hr  | 1.65 GB    |
  | KakaoTalk (of my own)              | 13,609       | 32 / 1                | 55.5 / 0.216        | 1.3 hr  | 6.01 GB    |
  | KakaoTalk (of my own, one speaker) | 6,805        | 32 / 1                | 92.5 / 0.292        | 0.6 hr  | 5.36 GB    |

  (Here, BLEU is calculated <u>_within teacher forcing_</u>, so it might not properly represent the real performance.)

<br/>

<br/>

## References

1. Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).
