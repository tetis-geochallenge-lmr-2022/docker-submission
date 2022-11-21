# Geochallenge docker submission

## 1. Tree description

```bash
.
├── Dockerfile
├── environment.yml
├── geoai
│   ├── all_event_roberta-base_typeless-True_mode_strict-True_batchsize-32_geonlplify-False_epoch-6.model
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── training_args.bin
│   │   └── vocab.json
│   └── input.jsonl
├── README.md
└── tetis-geochallenge-submit-1.py

```

## 2. Build docker image
```bash
sudo docker build -t geochallenge-test .
```

## 3. Run container 
```bash
sudo docker run \
-v "$(pwd)"/input.jsonl:/geoai/input.jsonl:ro \
-v "$(pwd)"/output.jsonl:/geoai/output.jsonl \
geochallenge-test
```
:warning: output.jsonl file must exisit and write permission must be granted to the docker user:
```bash
touch output.jsonl
chmod o+w output.jsonl
```

## 4. Code 
### 4.1 Main script
#### Transform BILOU to IOB for aggregating sub-token:
```python
example = "My name is Wolfgang and I live in New York City, US"

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(example)
print(ner_results)
# >>> [{'entity': 'B-LOC', 'score': 0.9750616, 'index': 9, 'word': 'ĠNew', 'start': 34, 'end': 37}, {'entity': 'I-LOC', 'score': 0.8160579, 'index': 10, 'word': 'ĠYork', 'start': 38, 'end': 42}, {'entity': 'L-LOC', 'score': 0.8318275, 'index': 11, 'word': 'ĠCity', 'start': 43, 'end': 47}, {'entity': 'U-LOC', 'score': 0.7664982, 'index': 13, 'word': 'ĠUS', 'start': 49, 'end': 51}]
```
- aggregate subtokens (but without convert BILOU into IOB:
```python
# aggregate subtokens (but without convert BILOU into IOB
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner_results = nlp(example)
print(ner_results)
# >>> [{'entity_group': 'LOC', 'score': 0.8955598, 'word': ' New York', 'start': 34, 'end': 42}, {'entity_group': 'LOC', 'score': 0.8318275, 'word': ' City', 'start': 43, 'end': 47}, {'entity_group': 'LOC', 'score': 0.7664982, 'word': ' US', 'start': 49, 'end': 51}]
```
- convert BILOU into IOB:
```python
# convert BILOU into IOB
nlp.model.config.id2label = {k: v.replace('L-', 'I-').replace('U-', 'B-') for k, v in nlp.model.config.id2label.items()}
# aggregate
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner_results = nlp(example)
print(ner_results)
# >>> [{'entity_group': 'LOC', 'score': 0.87431574, 'word': ' New York City', 'start': 34, 'end': 47}, {'entity_group': 'LOC', 'score': 0.7664982, 'word': ' US', 'start': 49, 'end': 51}]
```
#### Trouble with the pipeline tokenizer
- The model was trained with the option of the tokenizer `add_prefix_space=True` ([see code](https://github.com/tetis-geochallenge-lmr-2022/EDA/blob/0ebff5f3ce9af88515dd77374423279090c513de/common_tools/load_data_geochallenge.py#L38)) or `is_split_into_words=True` ([see similar bug report](https://github.com/huggingface/transformers/issues/15785))

- Better handle hashtag
### 4.2 Format checker
```bash
python3 format_checker.py --input_path geoai/input.jsonl  --output_path geoai/output.jsonl
```
## Troubleshooting
1. Not enough space. remove old containers and images
```bash
# list all containers (even those which are switch off)
sudo docker container ls --all
# remove container
sudo docker container rm [container_id]
# remove images
sudo docker image rm [image_id]
```

