'''
Eurosat is a satellite dataset with 13 bands in 10 m resolution and
64x64 images with ten classes.

I will do different tests on it.
Compare 64,32,16 resolutions
Compare different classifiers
Augmentering, olika

Unsupervised self-labeling
Look into contrastive labeling

Make json config for training
Make json config for evaluation

'''

import os,sys
from ai_helper import constants_dataset
from ai_helper import constants_ai_h
from ai_helper import dataset_load

from transformers import AutoFeatureExtractor, SwinForImageClassification
from datasets import load_dataset, load_metric
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from PIL import Image
import requests

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandAugment,
    Resize,
    ToTensor,
)


def trainer_get(training_args):
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    return trainer



def load_swin_model():
    # load model
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    return model


def training_args_get():
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
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=PUSH_TO_HUB,
        bf16=True,
    )
    return train_args


def transforms_train_eurosat(feature_extractor):
    train_transforms = Compose(
        [
            #Resize(feature_extractor.size),
            Resize(320),
            #RandomResizedCrop((feature_extractor.size)),
            RandomResizedCrop(256),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    return train_transforms


def transforms_val_eurosat(feat):
    val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )
    return val_transforms


def create_data_dir(data_dir):
    not_created_dir = True
    counter = 1
    check_if_exists_dir = data_dir
    while not_created_dir:
        if not(os.path.isdir(check_if_exists_dir)):
            not_created_dir = False
        else:
            counter += 1
            check_if_exists_dir = data_dir + '_' + str(counter)
    return check_if_exists_dir

# ToDo
def swin_json_load(path_config):
    swin_config = 0
    return swin_config

# ToDo
def swin_json_dump(config, path_config):
    pass


#  used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch




###################################         parameters          ###################################



MODEL_CHECKPOINT = r'microsoft/swinv2-tiny-patch4-window8-256'

MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
DIR_MODEL_CACHE = constants_dataset.DIR_MODEL_CACHE
MODEL_NAME_TRAINED = MODEL_NAME + '''-finetuned-eurosat'''
DIR_TRAINING_DATA = os.path.join(constants_ai_h.DIR_EXPERIMENTS_SWIN, MODEL_NAME_TRAINED)
DIR_DATASET = constants_dataset.DIR_DATASET_BASE_EUROSAT
TEST_SIZE = 0.1

EPOCHS = 1            # default 3
WARMUP_RATIO = 0.2   # default 0.1
BATCH_SIZE = 64       # default 32
lr = 1e-4             # default 5e-5 on batch size 32
PUSH_TO_HUB = False
TRAINER_PUSH_TO_HUB = False
GRADIENT_ACCUMULATION_STEPS = 1

words = MODEL_CHECKPOINT.split('/')
SAVE_DATA_DIR = create_data_dir(DIR_TRAINING_DATA)
print('DIR_TRAINING_DATA : ' + SAVE_DATA_DIR)

###################################################################################################

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)


#   preprocessing the data

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)


# load dataset
#dataset = load_dataset(DIR_DATASET, data_files="https://madm.dfki.de/files/sentinel/EuroSAT.zip", cache_dir=DIR_MODEL_CACHE)

from datasets import load_dataset

#data_dir = constants_dataset.DIR_DATASET_EUROSAT_RGB_16
data_dir = constants_dataset.DIR_DATASET_EUROSAT_RGB_32
dataset = load_dataset("imagefolder", data_dir=data_dir)


# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=TEST_SIZE)
train_ds = splits['train']
val_ds = splits['test']

train_transforms = transforms_train_eurosat(feature_extractor)
val_transforms = transforms_val_eurosat(feature_extractor)

print(feature_extractor.size)

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

#   ToDo    save datset to disk, load from disk
#   encoded_dataset.save_to_disk("path/of/my/dataset/directory")
#   from datasets import load_from_disk
#   reloaded_dataset = load_from_disk("path/of/my/dataset/directory")


metric = load_metric("accuracy")

print(dataset["train"].features)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


# load model
model = load_swin_model()
training_args = training_args_get()
trainer = trainer_get(training_args)

# train
train_results = trainer.train()
# rest is optional but nice to have
dir_save_model = os.path.join(constants_dataset.DIR_MODELS_SAVED_SWIN, MODEL_NAME_TRAINED)

# ToDo update directorys
trainer.save_model(dir_save_model)
trainer.log_metrics("train", metrics=train_results.metrics)
trainer.save_metrics("train", metrics=train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics=metrics)
trainer.save_metrics("eval", metrics=metrics)

if TRAINER_PUSH_TO_HUB:
    trainer.push_to_hub()

# save metrics and data


# config eval











# visualize data
#ml_helper_visualization.show_image_grid(images, 4, permutate=False)


# prepare data
#feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
#model = SwinForImageClassification.from_pretrained(MODEL_NAME, cache_dir=DIR_MODEL_CACHE)


# train model



# test model
#feature_extractor = AutoFeatureExtractor.from_pretrained("erikejw/swin-tiny-patch4-window7-224-finetuned-eurosat")
#model = AutoModelForImageClassification.from_pretrained("erikejw/swin-tiny-patch4-window7-224-finetuned-eurosat")


exit(0)

url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
image = Image.open(requests.get(url, stream=True).raw)

repo_name = "erikejw/swin-tiny-patch4-window7-224-finetuned-eurosat"

feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

# prepare image for the model
encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# forward pass
with torch.no_grad():
  outputs = model(**encoding)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


### quick way to run the model inference

#from transformers import pipeline
#pipe = pipeline("image-classification", "erikejw/swin-tiny-patch4-window7-224-finetuned-eurosat")
#print(pipe(image))

#pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)

# save model and test data



'''

from PIL import Image
import os, sys

path = "path_to_image"
dirs = os.listdir( path )
resolution = (192,256)
def resize():
    for item in dirs:
        if item != ".DS_Store":
            im = Image.open(path + '/' + item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize(resolution, Image.ANTIALIAS)
            print(path + '/resized_' + item)
            imResize.save(path + '/' + item, 'JPEG')

resize()



# Train and save results
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Evaluate on validation set
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)



from transformers import Trainer, TrainingArguments

batch_size = 16
# Defining training arguments (set push_to_hub to false if you don't want to upload it to HuggingFace's model hub)
training_args = TrainingArguments(
    f"swin-finetuned-food101",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

# Instantiate the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)



from transformers import SwinForImageClassification, Trainer, TrainingArguments

labels = ds['train'].features['label'].names

# initialzing the model
model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes = True,
)



from datasets import load_dataset

# loading the dataset
ds = load_dataset('food101')

# getting an example
ex = ds['train'][400]
print(ex)

# seeing the image
image = ex['image']
image.show()

# getting all the labels
labels = ds['train'].features['label']
print(labels)

# getting label of our example
print(labels.int2str(ex['label']))
'''









