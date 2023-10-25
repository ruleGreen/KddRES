# KddRES
This is the first Cantonese knowledge-driven Dialogue Dataset for REStaurant (KddRES) in Hong Kong, which grounds the information in multi-turn conversations to one specific restaurant. It contains 0.8k conversations which derive from 10 restaurants with various styles in different regions. In addition to that, we designed fine-grained slots and intents to better capture semantic information.

We have also prepared a version of Simplified Chinese.
- data_zh_hans: Simplified Chinese version
- data_zh_hant: Cantonese version
## Data
We introduce the concept of secondary slots to better record fine-grained slots’ information. A secondary slot refers to a slot that containing more detailed information. For example, when a user wants to know about a restaurant’s dishes, task-oriented dialogue system can not only tell the name of the dishes, but also the price of the dishes. User can also specifically ask about the price of a certain dish. Dishes and Dishes-price are the secondary slots in this scenario.

All secondary slots and their primary slots in KddRES(translated into English):
![image](figures/secondary_slot.png)

Data statistics:

![image](figures/statistic.png)

POSS is the proportion of dialogues that contain the secondary slots
## Experiment
We have implemented NLU(Natural Language Understanding) on KddRES with four models: BiLSTM, BiLSTM+CRF,BERTNLU and HierBERT. 
### Running
```
cd baseline
```
### BiLSTM
```
python scripts/get_BERT_word_embedding_for_a_dataset.py --in_files data/kddres/{train,valid,test} --output_word2vec local/word_embeddings/bert_768_cased_for_multilingual_es.txt --pretrained_tf_type bert --pretrained_tf_name hfl/chinese-bert-wwm-ext
```
```
bash run/kddres_Bilstm.sh
```
### BiLSTM+CRF
```
python scripts/get_BERT_word_embedding_for_a_dataset.py --in_files data/kddres/{train,valid,test} --output_word2vec local/word_embeddings/bert_768_cased_for_multilingual_es.txt --pretrained_tf_type bert --pretrained_tf_name hfl/chinese-bert-wwm-ext
```
```
bash run/kddres_Bilstm+crf.sh
```
### BERTNLU
```bash
bash run/kddres_bertnlu.sh
```

### HierBERT
```bash
bash run/kddres_bertcrf.sh
```

### Results
![image](figures/result.png)
## Citing

```
@misc{wang2020kddres,
      title={KddRES: A Multi-level Knowledge-driven Dialogue Dataset for Restaurant Towards Customized Dialogue System}, 
      author={Hongru Wang and Min Li and Zimo Zhou and Gabriel Pui Cheong Fung and Kam-Fai Wong},
      year={2020},
      eprint={2011.08772},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Copyright

The datasets are released for academic research only and it is free to researchers from educational or research institutions for non-commercial purposes. When downloading the dataset you agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
