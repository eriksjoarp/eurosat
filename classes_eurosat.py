'''
class for training swin models

ToDo
conf default for swin_config that is loaded and then changes can be made to it
json_load config
create base config json on disk
class training
class dataset


'''
import json
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
    def __init__(self, conf_json_path):
        self.config_path=conf_json_path
        self.config=[]

        # check if resources are available, then book resources
        # how should I do this, resource_config file?
        # load config file defaults
        ###############     params load from file later     ##################

        self.EPOCHS = 1
        self.lr = 2e-4
        self.MODEL_CHECKPOINT = r'microsoft/swinv2-tiny-patch4-window8-256'
        self.WEIGHT_DECAY = 0.0

        self.MODEL_NAME = self.MODEL_CHECKPOINT.split("/")[-1]
        self.DIR_MODEL_CACHE = c_d.DIR_MODEL_CACHE

        MODEL_ENDING = 'rgb-baseline_' + str(self.EPOCHS) + 'e_' + str(self.lr) + 'lr_' + str(self.WEIGHT_DECAY) + 'wd_'
        self.MODEL_NAME_TRAINED = self.MODEL_NAME + '''-finetuned-eurosat-''' + MODEL_ENDING

        self.TRAIN_BATCH_SIZE = dataset_load_helper.get_batchsize(self.MODEL_NAME_TRAINED)
        self.TRAIN_BATCH_SIZE = 16
        self.EVAL_BATCH_SIZE = self.TRAIN_BATCH_SIZE
        self.GRADIENT_ACCUMULATION_STEPS = int(160 / 2 / self.TRAIN_BATCH_SIZE)  # BATCH_SIZE * GPUs * GRAD_ACC = 128

        self.DATALOADER_NUM_WORKERS = 4
        self.WARMUP_RATIO = 0.25
        self.PUSH_TO_HUB = False
        self.TEST_SIZE = 0.1

        self.DATASET_SMALL = True
        self.DATA_DIR = c_d.DIR_DATASET_EUROSAT_RGB

        self.DIR_TRAINING_DATA = os.path.join(constants_ai_h.DIR_EXPERIMENTS_SWIN, self.MODEL_NAME_TRAINED)
        self.DIR_TRAINING_DATA = dataset_load_helper.create_data_dir(self.DIR_TRAINING_DATA)   # create unique directory
        self.SAVE_DATA_DIR = dataset_load_helper.create_data_dir(self.DIR_TRAINING_DATA)

        ######################################################################

        # check if gpus are available
        torch_help_functions.is_cuda_available()

        # extract features
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_CHECKPOINT)
        #self.normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        self.normalize = self.normalize_create()
        self.train_transforms = self.transforms_train_eurosat(self.feature_extractor)
        self.val_transforms = self.transforms_val_eurosat()

        # load transformers

        # load dataset
        self.dataset_loaded = self.dataset_load()

        # preprocess datasets
        self.dataset_loaded['train'].set_transform(self.preprocess_train)
        self.dataset_loaded['test'].set_transform(self.preprocess_val)
        self.dataset_loaded['val'].set_transform(self.preprocess_val)

        # load model
        self.model_swin = self.load_swin_model()

        # start training
        self.training_args_swin = self.training_args_get()
        self.trainer_swin = self.trainer_get(self.training_args_swin)

        # evaluate data

        # save metrics

        # release resources


    def train(self):
        self.train_results = self.trainer_swin.train()

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
        self.model = Swinv2ForImageClassification.from_pretrained(
            self.MODEL_CHECKPOINT,
            label2id=self.label2id,
            id2label=self.id2label,
            # use_auth_token=,           # huggingface login
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        return self.model

    def training_args_get(self):
        # config training
        train_args = TrainingArguments(
            # f"{MODEL_NAME}-finetuned-eurosat",
            self.SAVE_DATA_DIR,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            per_device_eval_batch_size=self.EVAL_BATCH_SIZE,
            num_train_epochs=self.EPOCHS,
            dataloader_num_workers=self.DATALOADER_NUM_WORKERS,
            warmup_ratio=self.WARMUP_RATIO,
            # weight_decay=WEIGHT_DECAY,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=self.PUSH_TO_HUB,
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
                self.normalize,
            ]
        )
        return train_transforms

    def transforms_val_eurosat(self):
        val_transforms = Compose(
            [
                Resize(self.feature_extractor.size),
                CenterCrop(self.feature_extractor.size),
                ToTensor(),
                self.normalize,
            ]
        )
        return val_transforms

    def compute_metrics_train(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        self.conf_matrix(labels, preds, show_matrix=False)
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

    def preprocess_train(self, example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.train_transforms(image.convert("RGB")) for image in example_batch["image"]]  # examples["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
        return example_batch

    def preprocess_val(self, example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def normalize_create(self):
        self.normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)

    # ToDo
    def swin_json_load(self):
        json.load(self.config['json'])

    # ToDo
    def swin_json_dump(self):
        json.dump(self.config)

    def map_labels(self):
        self.label2id, self.id2label = dict(), dict()

        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
                   'Residential', 'River', 'SeaLake']
        for i, label in enumerate(classes):
            self.label2id[label] = i
            self.id2label[i] = label

    def dataset_load(self):
        self.dataset = dataset_load.dataset_load_from_imagefolder(self.DATA_DIR, dataset_small=self.DATASET_SMALL)

    def metric_load(self, metric_list='accuracy'):
        self.metric = load_metric(metric_list)

    def data_save(self):
        dir_save_model = os.path.join(c_d.DIR_MODELS_SAVED_SWIN, self.MODEL_NAME_TRAINED)
        dir_eval = os.path.join(dir_save_model, 'eval')

        self.trainer.save_model(dir_save_model)
        self.trainer.log_metrics(dir_save_model, metrics=self.train_results.metrics)
        self.trainer.save_metrics(dir_save_model, metrics=self.train_results.metrics)
        self.trainer.save_state()

        metrics = self.trainer.evaluate()
        self.trainer.log_metrics(dir_eval, metrics=metrics)
        self.trainer.save_metrics(dir_eval, metrics=metrics)

        if self.PUSH_TO_HUB:
            self.trainer.push_to_hub()

        print('DIR_TRAINING_DATA  ' + self.DIR_TRAINING_DATA)

        print('dir_save_model  ' + dir_save_model)
        print('dir_eval  ' + dir_eval)

        self.training_args.do_train = False

        # move confusion matrix pngs to traindir
        dir_conf_matrix = os.path.join(dir_save_model, 'figs')

        # def copy_files_path(dir_from, dir_save, delete_old=False):
        erik_functions_files.copy_files_from_dir_to_dir(constants_ai_h.DIR_LOCAL_FIGS, dir_conf_matrix,
                                                        delete_old_src=True, delete_old_save=True)





