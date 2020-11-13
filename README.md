# DKN

This repository is a fork of the implementation of [DKN](https://dl.acm.org/citation.cfm?id=3186175) made by [@hwwang55](https://github.com/hwwang55) and published in this [repo](https://github.com/hwwang55/DKN).

I've upgraded the code to TensorFlow 2.3 and also added the option to use [SciBERT](https://github.com/allenai/scibert) as word embeddings. 

From the original repo
> DKN is a deep knowledge-aware network that takes advantage of knowledge graph representation in news recommendation.
The main components in DKN is a KCNN module and an attention module:
> - The KCNN module is to learn from semantic-level and knowledge-level representations of news jointly.
> The multiple channels and alignment of words and entities enable KCNN to combine information from heterogeneous sources.
> - The attention module is to model the different impacts of a user’s diverse historical interests on current candidate news.


### Files in the folder

- `data/`
  - `kg/`
    - `Fast-TransX`: an efficient implementation of TransE and its extended models for Knowledge Graph Embedding (from https://github.com/thunlp/Fast-TransX);
    - `kg.txt`: knowledge graph file;
    - `kg_preprocess.py`: pre-process the knowledge graph and output knowledge embedding files for DKN;
    - `prepare_data_for_transx.py`: generate the required input files for Fast-TransX;
  - `news/`
    - `news_preprocess.py`: pre-process the news dataset;
    - `raw_test.txt`: raw test data file;
    - `raw_train.txt`: raw train data file;
- `src/`: implementations of DKN.

> Note: Due to the pricacy policies of Bing News and file size limits on Github, the released raw dataset and the knowledge graph in this repository is only a small sample of the original ones reported in the paper.


### Format of input files
- **raw_train.txt** and **raw_test.txt**:  
  `user_id[TAB]news_title[TAB]label[TAB]entity_info`  
  for each line, where `news_title` is a list of words `w1 w2 ... wn`, and `entity_info` is a list of pairs of entity id and entity name: `entity_id_1:entity_name;entity_id_2:entity_name...`
- **kg.txt**:  
  `head[TAB]relation[TAB]tail`  
  for each line, where `head` and `tail` are entity ids and `relation` is the relation id.


### Required packages
The code has been tested running under Python 3.6.5. To run it you should install the requirements listed in `requirements.txt`.

### Running the code
```
$ cd data/news
$ python news_preprocess.py
$ cd ../kg
$ python prepare_data_for_transx.py
$ cd Fast-TransX/transE/ (note: you can also choose other KGE methods)
$ g++ transE.cpp -o transE -pthread -O3 -march=native
$ ./transE
$ cd ../..
$ python kg_preprocess.py
$ cd ../../src
$ python main.py --entity_embeddings="/home/ivania/DKN/data/kg/entity_embeddings_TransE_50.npy"
(note: use -h to check optional arguments)
```
