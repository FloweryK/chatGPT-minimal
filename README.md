# chatGPT-minimal

A minimal, standalone pytorch implementation of chatGPT.

This project provides:

- Base dataset class which only requires a `csv` file as its data source.
- Custom dataset class samples:
  - [Cornell movie-dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  - [Korean Chatbot data](https://github.com/songys/Chatbot_data)
  - KakaoTalk (downloaded from your own mobile phone or PC)
- Optimization options for the training:
  - Data Augmentation based on Doc2Vec
  - [Automatic Mixed Precision to FP16](https://pytorch.org/docs/stable/notes/amp_examples.html)
  - [Gradient Accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation)
  - Warmup scheduler
  

<br/>

<br/>

## How to prepare your dataset

##### Case 1: If you want to use your dataset as a csv file

- Prepare your dataset as a `csv` file with headers as 'Q' and 'A'. It might look like this:

  | Q                                                            | A                                                 |
  | ------------------------------------------------------------ | ------------------------------------------------- |
  | Hey, you said one out of fourteen million, we'd win, yeah? Tell me this is it. | If I tell you what happens, it won't happen.      |
  | Did she have any family?                                     | Yeah. Us.                                         |
  | Don't do anything stupid until I come back.                  | How can I? You're taking all the stupid with you. |
  | ...                                                          | ...                                               |

<br/>

##### Case 2: If you want to create your own dataset class

- You may use `ChatDatasetBase` and  implement `self.load_data` so that your dataset is stored in `self.data`. 

- `self.data` is a dictionary, and **MUST** have its elements as the following format:

  ```python
  self.data[answer_id] = {
      'id': answer_id,
      'text': [text],
      'question_id': question_id,
  }
  ```

- Here's a [sample code](dataset/movie_corpus.py) which loads [Cornell movie-dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset. You can find more in `dataset/` directory.

<br/>

<br/>

## How to train

```bash
$ python train.py -d {path-to-data}
```

| flag       | description         | example                         | default  |
| ---------- | ------------------- | ------------------------------- | -------- |
| -d, --data | path to the dataset | -d src/dataset/chatbot_data.csv | Required |

- the corresponding files will be stored in `runs/{datetime}_vocab={vocab_size}...`

<br/>

##### (Optional) Run tensorboard and monitor metircs

```bash
$ tensorboard --logdir=runs/
```

<br/>

<br/>

## How to talk to a trained model

```bash
$ python chatbot.py -v {path-to-vocab} -w {path-to-weight}
```

| flag         | description                           | example                          | default  |
| ------------ | ------------------------------------- | -------------------------------- | -------- |
| -v, --vocab  | path to the sentencepiece vocab model | -v runs/20230830_.../vocab.model | Required |
| -w, --weight | path to the trained weight            | -w runs/20230830_.../model_49.pt | Required |

```bash
$ python chatbot.py -v {path-to-vocab} -w {path-to-weight}
text: who are you?

[2, 187, 124, 16, 19991, 3] # encoded pieces of the input text
who are you?	# input text

[2, 8, 19984, 19981, 20, 7805, 19976, 3] # encoded pieces of the answer
i'm the creator.	# answer
```

<br/>

<br/>

## Performance Test Results

- config used for testing (everything is the same except `n_batch` and `n_accum` due to OOM)

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
  augment = True
  augment_topn = 10
  augment_threshold = 0.7
  
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
  
  (Here, BLEU is calculated <u>*within teacher forcing*</u>, so it might not properly represent the real performance.)

<br/>

<br/>

## References

1. Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).
