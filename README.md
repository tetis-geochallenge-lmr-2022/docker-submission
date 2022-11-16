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
sudo docker run geochallenge-test
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

