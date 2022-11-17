"""
Tetis Geochallenge sybmit 1
"""
__author__ = "Remy Decoupes, UMR TETIS, INRAE"
__credits__ = "Rémy DECOUPES"


from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from transformers import pipeline
import json
import pandas as pd
from tqdm import tqdm

def jsonstr_to_df(json_str):
    return pd.json_normalize(json.loads(json_str))

def nlp_results_to_location_mentions(entities):
    list_location_mentions = []
    for ent in entities:
        location_mention = {
            "text": ent["word"],
            "start_offset": ent["start"],
            "end_offset": ent["end"]
        }
        list_location_mentions.append(location_mention)
    return list_location_mentions

# Load checkpoint weight
label_encoding_dict_type_less = {"O": 0, "B-LOC": 1, "U-LOC": 2, "I-LOC": 3, "L-LOC": 4}
label_list_type_less = ["O", "B-LOC", "U-LOC", "I-LOC", "L-LOC"]
id2label = {i: label for i, label in enumerate(label_list_type_less)}
label2id = {v: k for k, v in id2label.items()}
models_saved_path = "./geoai/"
model_name = "all_event_roberta-base_typeless-True_mode_strict-True_batchsize-32_geonlplify-False_epoch-6.model"
model_path = models_saved_path + model_name
model = RobertaForTokenClassification.from_pretrained(model_path,
                                                      id2label=id2label,
                                                      label2id = label2id
                                                     )
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)


with open('geoai/input.jsonl', 'r') as json_file:
    json_list = list(json_file)

df = pd.concat(map(jsonstr_to_df, json_list))

# transforms bilou format into IOB in order to do an aggregation
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
nlp.model.config.id2label = {k: v.replace('L-', 'I-').replace('U-', 'B-') for k, v in nlp.model.config.id2label.items()}

tqdm.pandas()
df["predicted"] = df["text"].progress_apply(nlp)
df["location_mentions"] = df["predicted"].progress_apply(nlp_results_to_location_mentions)

output_jsonl = df[["tweet_id", "location_mentions"]].to_json(orient="records", lines=True)
f_output_jsonl = open("./geoai/output.jsonl", "w")
f_output_jsonl.write(output_jsonl)
f_output_jsonl.close()

"""
TODO:
 - use jsonl file as an input : ok
 - apply pipeline on tweet : ok
 - remove subtokens : ok
 - start building a jsonl output file : ok
 - check output.jsonl format 
"""