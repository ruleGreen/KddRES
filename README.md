# KddRES
This is the first Cantonese knowledge-driven Dialogue Dataset for REStaurant (KddRES) in Hong Kong, which grounds the information in multi-turn conversations to one specific restaurant.It contains 0.8k conversations which derive from 10 restaurantswith various styles in different regions. In addition to that, we designed fine-grained slots and intents to better capture semantic information.
We have also prepared a version of Simplified Chinese.
## Data


## Experiment
We implement 
you can 
### Code
```
cd baseline
```
BiLSTM:
```
python scripts/get_BERT_word_embedding_for_a_dataset.py --in_files data/kddres/{train,valid,test} --output_word2vec local/word_embeddings/bert_768_cased_for_multilingual_es.txt --pretrained_tf_type bert --pretrained_tf_name hfl/chinese-bert-wwm-ext
```
```
bash run/kddres_Bilstm.sh
```
BiLSTM+CRF:
```
python scripts/get_BERT_word_embedding_for_a_dataset.py --in_files data/kddres/{train,valid,test} --output_word2vec local/word_embeddings/bert_768_cased_for_multilingual_es.txt --pretrained_tf_type bert --pretrained_tf_name hfl/chinese-bert-wwm-ext
```
```
bash run/kddres_Bilstm+crf.sh
```
BERT
```
bash run/kddres_bert.sh
```
BERT+crf
```
bash run/kddres_bertcrf.sh
```
### Results

## Citing
