# chatGPT-minimal

A minimal, standalone pytorch implementation of chatGPT.

<br/>

This project contains:

- A base dataset class that uses [sentencepiece](https://github.com/google/sentencepiece) as its tokenizer, so that you don't have to tokenize your own data.
- Some sample dataset classes (in `dataset/`):
  - [Cornell movie-dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  - KakaoTalk (downloaded from your own mobile phone or PC)
  - [Korean Chatbot data](https://github.com/songys/Chatbot_data)
- Some optimization options for the training:
  - [Automatic Mixed Precision to FP16](https://pytorch.org/docs/stable/notes/amp_examples.html)
  - Gradient Accumulation
  - A warmup scheduler

<br/>

<br/>

## How to train

1. Prepare your dataset as a `csv` file with headers as 'Q' and 'A'. It might look like this:

   | Q                                                            | A                                                 |
   | ------------------------------------------------------------ | ------------------------------------------------- |
   | Hey, you said one out of fourteen million, we'd win, yeah? Tell me this is it. | If I tell you what happens, it won't happen.      |
   | Did she have any family?                                     | Yeah. Us.                                         |
   | Don't do anything stupid until I come back.                  | How can I? You're taking all the stupid with you. |
   | ...                                                          | ...                                               |

   (Optional) Or, you can prepare your own dataset using `ChatDatasetBase`.

   - If so, you sholud implement `self.load_data` so that your dataset is stored in `self.data`. `self.data` is a dictionary, and **MUST** have its elements as the following format:

     ```python
     self.data[answer_id] = {
         'id': answer_id,
         'text': [text],
         'question_id': question_id,
     }
     ```

   - Here's a sample code which loads [Cornell movie-dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset.

     ```python
     from dataset.base import ChatDatasetBase
      
     
     class MovieCorpusDataset(ChatDatasetBase):
         def load_data(self, path_data):
             with open(path_data, 'r', encoding='iso-8859-1') as f:
                 for line in f:
                     # load line as json
                     obj = json.loads(line)
     
                     # make converstaion data
                     text = obj['text'].strip().lower()
                     answer_id = obj['id']
                     question_id = obj['reply-to']
     
                     self.data[answer_id] = {
                         'id': answer_id,
                         'text': [text],
                         'question_id': question_id,
                     }
      
     # dataset loading
     dataset = MovieCorpusDataset(path_data='utterances.jsonl')
     ```

2. Run training. 

   ```bash
   $ python train.py -d {path-to-data} -v {path-to-vocab}
   ```

   | flag        | description                           | example                                      | default               |
   | ----------- | ------------------------------------- | -------------------------------------------- | --------------------- |
   | -d, --data  | path to the dataset                   | -d src/dataset/movie-corpus/utterances.jsonl | required              |
   | -v, --vocab | path to the sentencepiece vocab model | -v src/vocab/movie-corpus/vocab.model        | `src/vocab/tmp.model` |

   - If you specify the vocab model path (`-v`):

     - if you don't have a pre-trained vocab model at the path, then `ChatDatasetBase` will automatically train a vocab model based on `sentencepiece` and save the model into the vocab model path.
     - if you have a pre-trained vocab model at the path, then `ChatDatasetbase` will use it to encode texts.

   - Otherwise, `ChatDatasetBase` with train and save the vocab model into the default path.

3. (Optional) You may run tensorboard if you want:

   ```bash
   $ tensorboard --logdir=runs/
   ```

   

<br/>

<br/>

## How to talk to a trained model

```bash
$ python chatbot.py -v {path-to-vocab} -w {path-to-weight}
```

| flag         | description                           | example                               | default  |
| ------------ | ------------------------------------- | ------------------------------------- | -------- |
| -v, --vocab  | path to the sentencepiece vocab model | -v src/vocab/movie-corpus/vocab.model | required |
| -w, --weight | path to the trained weight            | -w weights/model_49.pt                | required |

<br/>

<br/>

## References

1. Vaswani, Ashish, et al. "Attention is all you need." _Advances in neural information processing systems_ 30 (2017).
