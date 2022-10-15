'''
class for training swin models

ToDo
conf default for swin_config that is loaded and then changes can be made to it
json_load config
create base config json on disk
class training
class dataset


'''


import os,sys
from ai_helper import constants_dataset as c_d
from ai_helper import constants_ai_h
from ai_helper import dataset_load
from ai_helper import dataset_load_helper
from ai_helper import torch_help_functions

from helper import  erik_functions_files
import eurosat_helper
import matplotlib.pyplot as plt

from transformers import AutoFeatureExtractor, SwinForImageClassification
from datasets import load_dataset, load_metric
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, Swinv2ForImageClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns

import torch, torchvision
from PIL import Image
import requests
import time

from ai_helper import constants_ai_h as c_ai


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandAugment,
    AutoAugment,
    ColorJitter,
    RandomPerspective,
    RandomRotation,
    Resize,
    ToTensor,
)





class EurosatDataset:
    def __init__(self, conf_dataset_json):
        self.conf_dataset=conf_dataset_json  # json load from path



# ToDo fix conf_json

class SwinTraining:
    def __init__(self, conf_json):
        self.config_train=conf_json


    def train(self):
        pass

    def eval(self):
        pass

    def conf_matrix(self, y, y_pred, show_matrix=False):
        # y_pred = logreg.predict(X)  # Get the confusion matrix
        cf_matrix = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
        if show_matrix:
            plt.show()
        else:
            print(cf_matrix)
        save_filename = dataset_load_helper.get_filename_unique('figs', 'confusion_matrix_')
        plt.imsave(save_filename)

    def trainer_get(self, training_args):
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=self.dataset[c_d.DATASET_TRAIN],
            eval_dataset=self.dataset[c_d.DATASET_VAL],
            tokenizer=self.feature_extractor,
            compute_metrics=self.compute_metrics_train,
            data_collator=self.collate_fn,
        )
        return trainer

    def load_swin_model(self):
        # load model
        # model = AutoModelForImageClassification.from_pretrained(
        model = Swinv2ForImageClassification.from_pretrained(
            MODEL_CHECKPOINT,
            label2id=label2id,
            id2label=id2label,
            # use_auth_token=,           # huggingface login
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        return model

    def training_args_get(self):
        # config training
        train_args = TrainingArguments(
            # f"{MODEL_NAME}-finetuned-eurosat",
            SAVE_DATA_DIR,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            warmup_ratio=WARMUP_RATIO,
            # weight_decay=WEIGHT_DECAY,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=PUSH_TO_HUB,
            seed=1973,
            bf16=True,
            # gradient_checkpointing=True,       # super slow
            # auto_find_batch_size=True,         # is used with pip install accelerate
            # hub_token=,                        # when push to hub
        )
        return train_args

    def transforms_train_eurosat(self, feature_extractor):
        train_transforms = Compose(
            [
                Resize(feature_extractor.size),
                # Resize(320),
                # RandomResizedCrop(feature_extractor.size),
                # RandomResizedCrop(256),
                # RandomHorizontalFlip(),
                # RandAugment(),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )
        return train_transforms

    def transforms_val_eurosat(self):
        val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )
        return val_transforms

    def compute_metrics_train(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        conf_matrix(labels, preds, show_matrix=False)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    #  used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels
    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch[
                "image"]]  # examples["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
        return example_batch

    def preprocess_val(self, example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


