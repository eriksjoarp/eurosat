'''
class for training swin models

ToDo
class dataset from imagefolder
class logging
class eval

confusion_matrix_training and evaluation : differentiate
check eval saving metrics etc

transforms: create ablation easily,
    sequence of
    2 out of X randomly

cosine annealing lr

logger
conf_matrix better colors for smaller values

load_trained wights to continue training or for evaluation

'''

import json
import os,sys
import random
import pandas as pd
import numpy as np

from ai_helper import constants_dataset as c_d
from ai_helper import constants_ai_h as c_ai_h
from ai_helper import dataset_load
from ai_helper import dataset_load_helper
from ai_helper import torch_help_functions
from ai_helper import constants_training as c_t
from ai_helper import ml_helper_visualization

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
    RandomVerticalFlip,
    RandomEqualize,
    Resize,
    ToTensor,
)


'''
class EurosatDataset:
    def __init__(self, conf_dataset_json):
        self.conf_dataset=conf_dataset_json  # json load from path
'''


class SwinTraining:
    def __init__(self, config_json=None, config_json_path=None, config_json_dataset=None, config_json_dataset_path=None):     # start with reading the baseconfig and change it and save to disc before initializing this class
        if config_json == None:
            self.config_path=config_json_path
            self.c = erik_functions_files.json_load(self.config_path)
        else:
            self.c = config_json

        if config_json_dataset == None:
            self.config_path_dataset=config_json_dataset_path
            self.c_d = erik_functions_files.json_load(self.config_path_dataset)
        else:
            self.c_d = config_json_dataset

        self.initialize_config_values()

        print(self.c)
        print(self.c_d)

        # check if resources are available, then book resources, GPU


        # check if gpus are available
        torch_help_functions.is_cuda_available()

        # extract features
        self.feature_extractor = self.feature_extractor_swin()

        # load transformers
        self.normalize = self.normalize_create()
        self.train_transforms = self.transforms_train_eurosat()
        self.val_transforms = self.transforms_val_eurosat()

        # load dataset
        self.datasets = self.dataset_load()

        # map labels
        self.label2id, self.id2label = self.map_labels()

        # preprocess datasets
        self.datasets['train'].set_transform(self.preprocess_train)
        self.datasets['test'].set_transform(self.preprocess_val)
        self.datasets['val'].set_transform(self.preprocess_val)

        # load model
        self.model_swin = self.load_swin_model()

        # start training
        self.training_args_swin = self.training_args_get()
        self.trainer_swin = self.trainer_get()

        #self.train_results = self.train()

        # evaluate data

        # save metrics
        #self.data_save()

        # release resources


    # train the model
    def train(self):
        _train_results = self.trainer_swin.train()
        self._train_results = _train_results
        return _train_results


    def eval(self):
        pass


    # ToDo
    # run this after changing the config to recalculate dependency values
    def initialize_config_values(self):
        self.c[c_t.S2_MODEL_NAME_TRAINED] = self.c[c_t.S2_MODEL_CHECKPOINT].split("/")[-1]    # remove / from checkpoint

        if self.c[c_t.S2_MODEL_NAME_TRAINED_PRE_CHOSEN] == 'False':
            _MODEL_ENDING = 'rgb-baseline_' + str(self.c[c_t.S2_TRAIN_ARGS_NUM_TRAIN_EPOCHS]) + 'e_' + str(self.c[c_t.S2_TRAIN_ARGS_LR]) + \
                            'lr_' + str(self.c[c_t.S2_TRAIN_ARGS_WEIGHT_DECAY]) + 'wd_resize-256_' + str(random.randint(0,1000))
            self.c[c_t.S2_MODEL_NAME_TRAINED] = self.c[c_t.S2_MODEL_NAME] + '''-finetuned-eurosat-''' + _MODEL_ENDING
        else:
            self.c[c_t.S2_MODEL_NAME_TRAINED] = self.c[c_t.S2_MODEL_NAME_TRAINED_PRE_CHOSEN]

        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                        'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

        self._dir_save_model = os.path.join(c_d.DIR_MODELS_SAVED_SWIN, self.c[c_t.S2_MODEL_NAME_TRAINED], 'figs')
        os.makedirs(self._dir_save_model, exist_ok=True)

        # retrieve batch sizes to use ToDo fix if needed
        #self.c[c_t.S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE] = dataset_load_helper.get_batchsize(self.c[c_t.S2_MODEL_NAME_TRAINED])
        #self.c[c_t.S2_TRAIN_ARGS_PER_DEVICE_EVAL_BATCH_SIZE] = self.c[c_t.S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE]

        self.c_d[c_t.DATASET_DIR_TRAINING_DATA_BASE] = os.path.join(c_ai_h.DIR_EXPERIMENTS_SWIN_LARGE, self.c[c_t.S2_MODEL_NAME_TRAINED])

        # ToDo probably wrong, what should it be? ERROR
        #self.c_d[c_t.DATASET_DIR_TRAINING_DATA] = dataset_load_helper.create_data_dir(self.c_d[c_t.DATASET_DIR_TRAINING_DATA_BASE])           # create unique directory
        self.c[c_t.S2_SAVE_DATA_DIR] = dataset_load_helper.create_data_dir(self.c_d[c_t.DATASET_DIR_TRAINING_DATA_BASE])

        # validate the config
        self.config_validate()


    def config_validate(self):      # ToDo validate the config so it makes sense
        pass

    # extract features from the pretrained model
    def feature_extractor_swin(self):
        _feature_extractor = AutoFeatureExtractor.from_pretrained(self.c[c_t.S2_MODEL_CHECKPOINT])
        return _feature_extractor

    # calculate confusion matrix
    def conf_matrix(self, y, y_pred, show_matrix=False, save_matrix=True, do_corr_matrix=True):
        #def conf_matrix(y, y_pred, path_save, filename_save, show_matrix=False, save_matrix=True, x_labels=None, y_labels=None, do_corr_matrix=True):
        conf_matrix = ml_helper_visualization.conf_matrix(y=y, y_pred=y_pred, path_save=self._dir_save_model, filename_save='confusion_matrix_.png', show_matrix=show_matrix,
                                                          save_matrix=save_matrix, x_labels=self.classes, y_labels=self.classes, do_corr_matrix=False)
        corr_matrix = ml_helper_visualization.conf_matrix(y=y, y_pred=y_pred, path_save=self._dir_save_model, filename_save='corr_matrix_.png', show_matrix=show_matrix,
                                                          save_matrix=save_matrix, x_labels=self.classes, y_labels=self.classes, do_corr_matrix=True)
        return conf_matrix, corr_matrix


    # define the huggingface trainer
    def trainer_get(self):
        _training_args = self.training_args_swin
        _trainer_swin = Trainer(
            self.model_swin,
            _training_args,
            train_dataset=self.datasets[c_d.DATASET_TRAIN],
            eval_dataset=self.datasets[c_d.DATASET_VAL],
            tokenizer=self.feature_extractor,
            compute_metrics=self.compute_metrics_train,
            data_collator=self.collate_fn,
        )
        return _trainer_swin

    def load_swin_model(self):
        # load model
        # model = AutoModelForImageClassification.from_pretrained(
        _model = Swinv2ForImageClassification.from_pretrained(
            self.c[c_t.S2_MODEL_CHECKPOINT],
            label2id=self.label2id,
            id2label=self.id2label,
            # use_auth_token=,           # huggingface login
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        return _model


    def training_args_get(self):
        # config training
        _train_args = TrainingArguments(
            # f"{MODEL_NAME}-finetuned-eurosat",
            self.c[c_t.S2_SAVE_DATA_DIR],
            remove_unused_columns=False,
            evaluation_strategy=self.c[c_t.S2_TRAIN_ARGS_EVALUATION_STRATEGY],
            save_strategy=self.c[c_t.S2_TRAIN_ARGS_SAVE_STRATEGY],
            learning_rate=self.c[c_t.S2_TRAIN_ARGS_LR],
            per_device_train_batch_size=self.c[c_t.S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE],
            gradient_accumulation_steps=self.c[c_t.S2_TRAIN_ARGS_GRADIENT_ACCUMULATION_STEPS],
            per_device_eval_batch_size=self.c[c_t.S2_TRAIN_ARGS_PER_DEVICE_EVAL_BATCH_SIZE],
            num_train_epochs=self.c[c_t.S2_TRAIN_ARGS_NUM_TRAIN_EPOCHS],
            dataloader_num_workers=self.c[c_t.S2_TRAIN_ARGS_DATALOADER_NUM_WORKERS],
            warmup_ratio=self.c[c_t.S2_TRAIN_ARGS_WARMUP_RATIO],
            # weight_decay=WEIGHT_DECAY,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            optim='adamw_torch',
            push_to_hub=self.c[c_t.S2_TRAIN_ARGS_PUSH_TO_HUB],
            seed=1973,
            bf16=True,
            # gradient_checkpointing=True,       # super slow
            # auto_find_batch_size=True,         # is used with pip install accelerate
            # hub_token=,                        # when push to hub
        )
        return _train_args

    def transforms_train_eurosat(self):
        _train_transforms = Compose(
            [
                Resize(self.feature_extractor.size),
                #Resize(320),
                # RandomResizedCrop(feature_extractor.size),
                # RandomResizedCrop(256),
                #RandomHorizontalFlip(),
                #RandomVerticalFlip(),
                # RandAugment(),
                # AutoAugment,
                # ColorJitter,
                #RandomPerspective,
                RandomRotation(30),
                RandomVerticalFlip(),
                #RandomEqualize,
                CenterCrop(self.feature_extractor.size),
                ToTensor(),
                self.normalize,
            ]
        )
        return _train_transforms

    def transforms_val_eurosat(self):
        _val_transforms = Compose(
            [
                Resize(self.feature_extractor.size),
                CenterCrop(self.feature_extractor.size),
                ToTensor(),
                self.normalize,
            ]
        )
        return _val_transforms

    def compute_metrics_train(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        self.conf_matrix(labels, preds, show_matrix=False, save_matrix=True)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    #  used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels
    def collate_fn(self, examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {'pixel_values': pixel_values, 'labels': labels}

    # preprocess training samples before the training by applying the train_transform
    def preprocess_train(self, example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.train_transforms(image.convert("RGB")) for image in example_batch["image"]]  # examples["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
        return example_batch

    # preprocess training samples before the training by applying the eval_transform
    def preprocess_val(self, example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # calculate the normalized values to use as transform
    def normalize_create(self):
        _normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        return _normalize

    # create the labelmapping between index and label_class
    def map_labels(self):
        _label2id, _id2label = dict(), dict()

        for i, label in enumerate(self.classes):
            _label2id[label] = i
            _id2label[i] = label
        return _label2id, _id2label

    # load the dataset
    def dataset_load(self):
        _datasets = dataset_load.dataset_load_from_imagefolder(self.c_d[c_t.DATASET_PATH_DATASET], dataset_small=self.c_d[c_t.DATASET_SMALL])
        # map labels
        self.map_labels()
        return _datasets

    # load the metric used for training
    def metric_load(self, metric_list='accuracy'):
        _metric = load_metric(metric_list)
        return _metric

    # dave data after training
    def data_save(self):
        _dir_save_model = os.path.join(c_d.DIR_MODELS_SAVED_SWIN, self.c[c_t.S2_MODEL_NAME_TRAINED])
        _dir_eval = os.path.join(_dir_save_model, 'eval')

        self.trainer_swin.save_model(_dir_save_model)
        self.trainer_swin.log_metrics(_dir_save_model, metrics=self._train_results.metrics)
        self.trainer_swin.save_metrics(_dir_save_model, metrics=self._train_results.metrics)
        self.trainer_swin.save_state()

        eval_metrics = self.trainer_swin.evaluate()
        self.trainer_swin.log_metrics(_dir_eval, metrics=eval_metrics)
        self.trainer_swin.save_metrics(_dir_eval, metrics=eval_metrics)

        if self.c[c_t.S2_TRAIN_ARGS_PUSH_TO_HUB]:
            self.trainer_swin.push_to_hub()

        print('DIR_TRAINING_DATA  ' + self.c_d[c_t.DATASET_DIR_TRAINING_DATA])

        print('dir_save_model  ' + _dir_save_model)
        print('dir_eval  ' + _dir_eval)

        self.training_args_swin.do_train = False

        # move confusion matrix pngs to traindir
        #dir_conf_matrix = os.path.join(_dir_save_model, 'figs')

        # def copy_files_path(dir_from, dir_save, delete_old=False):
        #erik_functions_files.copy_files_from_dir_to_dir(c_ai_h.DIR_LOCAL_FIGS, dir_conf_matrix, delete_old_src=True, delete_old_save=True)

