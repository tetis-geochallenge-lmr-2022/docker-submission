import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from tqdm.notebook import tqdm
tqdm.pandas()
from transformers import RobertaForTokenClassification, RobertaModel, RobertaTokenizerFast
from transformers import pipeline

def infer(list_tokens, model):
    """
    Made a prediction from a sentence in a shape of list of token (BILOU files)
    """
    inputs = tokenizer(list_tokens, truncation=True, is_split_into_words=True, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

def get_label_from_infer(list_tokens, model):
    """
    Get the label name predicted for each token (taking care of subword from roberta tokenizer)
    """
    try:
        softmax = nn.Softmax(dim=-1)
        outputs = infer(list_tokens, model)
        predictions = softmax(outputs.logits) # compute softmax
        labelled_predicted = torch.max(predictions, -1)[1][0] # get the label id for which the softmax is the greater
        label_names_predicted = []
        # Because of words are split into subword when they are out of Roberta vocabulary
        # so we detect those subwords and remove the label predicted associated
        label_ids = []
        tokenized_inputs = tokenizer(list_tokens, truncation=True, is_split_into_words=True, return_tensors="pt")
        words_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for j, token in enumerate(words_ids):
            if token is None: # we remove the CLF and SEP tokens
                pass
            elif token != previous_word_idx: # it's a new word (in the Roberta vocabulary) not a sub word
                label_ids.append(int(labelled_predicted[j]))
            else:
                pass
            previous_word_idx = token
            # labels.append(label_ids)
        labelled_predicted = label_ids
        # end subword labels remoging

        for label in labelled_predicted: # convert id to label names
            label_names_predicted.append(model.config.id2label[int(label)])
        return label_names_predicted
    except:
        np.nan


models_saved_path = "/geoai/"
label_encoding_dict_type_less = {"O": 0, "B-LOC": 1, "U-LOC": 2, "I-LOC": 3, "L-LOC": 4}
label_list_type_less = ["O", "B-LOC", "U-LOC", "I-LOC", "L-LOC"]
id2label = {i: label for i, label in enumerate(label_list_type_less)}
label2id = {v: k for k, v in id2label.items()}
model_name = "all_event_roberta-base_typeless-True_mode_strict-True_batchsize-32_geonlplify-False_epoch-6.model"
model_path = models_saved_path + model_name
model = RobertaForTokenClassification.from_pretrained(model_path,
                                                      id2label=id2label,
                                                      label2id = label2id
                                                     )
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

# With pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)

# with model outputs
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
