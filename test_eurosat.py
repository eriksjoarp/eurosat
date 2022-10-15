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

ConfusionMatrix

Automatiskt hyper_parameter_search:
    k-fold cross validation,    split data once more

Augment: Tiny, small        one of each, then combine 2 or three in random order
RandAugment
ColorJitter
RandomRotation(90)
RandomPerspective
RandomHorizontalFlip
RandomResizedCrop

lr: 5 olika
WEIGHT_DECAY:
Batch_size total with lr:
adam_beta1
adam_beta2
adam_epsilon
lr_scheduler_type
optim

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


# ToDo
def params_swin():
    params_swin_dict = {}

    # project
    # training_args
    # training


def conf_matrix(y, y_pred, show_matrix=False):
    #y_pred = logreg.predict(X)  # Get the confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(15, 10))
    svm = sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
    if show_matrix:
        plt.show()
    else:
        print(cf_matrix)
    save_filename = dataset_load_helper.get_filename_unique('figs', 'confusion_matrix_.png')
    plt.savefig(save_filename)
    #svm.savefig(save_filename, dpi=400)




def trainer_get(training_args):
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset[c_d.DATASET_TRAIN],
        eval_dataset=dataset[c_d.DATASET_VAL],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics_train,
        data_collator=collate_fn,
    )
    return trainer


def load_swin_model():
    # load model
    #model = AutoModelForImageClassification.from_pretrained(
    model = Swinv2ForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        label2id=label2id,
        id2label=id2label,
        #use_auth_token=,           # huggingface login
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
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        warmup_ratio=WARMUP_RATIO,
        #weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=PUSH_TO_HUB,
        seed=1973,
        bf16=True,
        #gradient_checkpointing=True,       # super slow
        #auto_find_batch_size=True,         # is used with pip install accelerate
        #hub_token=,                        # when push to hub
    )
    return train_args



def transforms_train_eurosat(feature_extractor):
    train_transforms = Compose(
        [
            Resize(feature_extractor.size),
            #Resize(320),
            #RandomResizedCrop(feature_extractor.size),
            #RandomResizedCrop(256),
            #RandomHorizontalFlip(),
            #RandAugment(),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )
    return train_transforms


def transforms_val_eurosat():
    val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )
    return val_transforms


def compute_metrics_train(pred):
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
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]]  # examples["pixel_values"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch



#CUDA_LAUNCH_BLOCKING=1
'''
MODEL_CHECKPOINT = r'microsoft/swinv2-small-patch4-window8-256'
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = transforms_train_eurosat(feature_extractor)
val_transforms = transforms_val_eurosat()
'''

###################################         parameters          ###################################

#MODEL_CHECKPOINT = r'microsoft/swinv2-tiny-patch4-window8-256'
MODEL_CHECKPOINT = r'microsoft/swinv2-tiny-patch4-window16-256'

#MODEL_CHECKPOINT = r'microsoft/swinv2-small-patch4-window8-256'
#MODEL_CHECKPOINT = r'microsoft/swinv2-small-patch4-window16-256'

#MODEL_CHECKPOINT = r'microsoft/swinv2-base-patch4-window8-256'
#MODEL_CHECKPOINT = r'microsoft/swinv2-base-patch4-window16-256'
#MODEL_CHECKPOINT = r'microsoft/swinv2-base-patch4-window12-192-22k'
#MODEL_CHECKPOINT = r'microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft'

#MODEL_CHECKPOINT = r'microsoft/swinv2-large-patch4-window12-192-22k'
#MODEL_CHECKPOINT = r'microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft'

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = transforms_train_eurosat(feature_extractor)
val_transforms = transforms_val_eurosat()


if __name__=='__main__':
    # CUDA_LAUNCH_BLOCKING=1

    MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
    DIR_MODEL_CACHE = c_d.DIR_MODEL_CACHE

    #   do autoaugment, colorjitter, RandAugment
    EPOCHS = 1  # default 3
    lr = 2e-4  # default 5e-5 on batch size 32
    WEIGHT_DECAY = 0

    MODEL_ENDING = 'rgb-baseline_' + str(EPOCHS) + 'e_' + str(lr) + 'lr_' + str(WEIGHT_DECAY) + 'wd_'
    MODEL_NAME_TRAINED = MODEL_NAME + '''-finetuned-eurosat-''' + MODEL_ENDING

    DIR_TRAINING_DATA = os.path.join(constants_ai_h.DIR_EXPERIMENTS_SWIN, MODEL_NAME_TRAINED)

    DATA_DIR = c_d.DIR_DATASET_EUROSAT_RGB
    DATASET_SMALL = c_d.DATASET_SMALL
    #DATASET_SMALL = False

    BATCH_SIZE = dataset_load_helper.get_batchsize(MODEL_NAME_TRAINED)

    WARMUP_RATIO = 0.25         # default 0.1
    #BATCH_SIZE = 32             # default 32
    PUSH_TO_HUB = False
    TRAINER_PUSH_TO_HUB = False
    GRADIENT_ACCUMULATION_STEPS = int(160 / 2 / BATCH_SIZE)     # BATCH_SIZE * GPUs * GRAD_ACC = 128

    DATALOADER_NUM_WORKERS = 2
    #WEIGHT_DECAY = 0.002
    TEST_SIZE = 0.1

    words = MODEL_CHECKPOINT.split('/')
    SAVE_DATA_DIR = dataset_load_helper.create_data_dir(DIR_TRAINING_DATA)
    print('DIR_TRAINING_DATA : ' + SAVE_DATA_DIR)

    ###################################################################################################

    # gpu found
    torch_help_functions.is_cuda_available()

    # create feature_extractor
    #feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    #   preprocessing the data
    #normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    # load dataset
    #dataset = load_dataset(DIR_DATASET, data_files="https://madm.dfki.de/files/sentinel/EuroSAT.zip", cache_dir=DIR_MODEL_CACHE)
    #def dataset_load_from_imagefolder(dir_dataset_base, dataset_small=False):

    #dataset = dataset_load.dataset_load_from_imagefolder(DATA_DIR, '_medium')
    dataset = dataset_load.dataset_load_from_imagefolder(DATA_DIR, dataset_small=DATASET_SMALL)
    print(feature_extractor.size , 'feature_extractor.size')

    # preprocess datasets
    dataset["train"].set_transform(preprocess_train)
    dataset["test"].set_transform(preprocess_val)
    dataset["val"].set_transform(preprocess_val)

    #   ToDo    save datset to disk, load from disk
    #   encoded_dataset.save_to_disk("path/of/my/dataset/directory")
    #   from datasets import load_from_disk
    #   reloaded_dataset = load_from_disk("path/of/my/dataset/directory")

    metric = load_metric("accuracy")

    label2id, id2label = dict(), dict()
    #for i, label in enumerate(dataset["train"].classes):
    #for i, label_ in enumerate(dataset["train"].features["label"]):

    #for i in range(10): ToDo try again
    #    label = dataset["train"].features["label"][i]
    #    label2id[label] = i
    #    id2label[i] = label

    # labels = ds['train'].features['label'].names
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    for i, label in enumerate(classes):
        label2id[label] = i
        id2label[i] = label

    # load model
    model = load_swin_model()
    training_args = training_args_get()
    trainer = trainer_get(training_args)

    # train
    train_results = trainer.train()

    # rest is optional but nice to have
    dir_save_model = os.path.join(c_d.DIR_MODELS_SAVED_SWIN, MODEL_NAME_TRAINED)
    dir_eval = os.path.join(dir_save_model, 'eval')
    #os.makedirs(dir_eval, exist_ok=True)
    #os.makedirs(dir_save_model, exist_ok=True)

    trainer.save_model(dir_save_model)
    trainer.log_metrics(dir_save_model, metrics=train_results.metrics)
    trainer.save_metrics(dir_save_model, metrics=train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics(dir_eval, metrics=metrics)
    trainer.save_metrics(dir_eval, metrics=metrics)

    if TRAINER_PUSH_TO_HUB:
        trainer.push_to_hub()

    print('DIR_TRAINING_DATA  ' + DIR_TRAINING_DATA)

    print('dir_save_model  ' + dir_save_model)
    print('dir_eval  ' + dir_eval)

    training_args.do_train = False

    # move confusion matrix pngs to traindir
    dir_conf_matrix = os.path.join(dir_save_model, 'figs')
    erik_functions_files.copy_files_path('figs', dir_conf_matrix, delete_old=True)


    exit(0)

    # confusion matrix

    #logreg = LogisticRegression(C=1e5)
    #logreg.fig(X, y)  # Generate predictions with the model using our X values
    y_pred = logreg.predict(X)  # Get the confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)
    print(cf_matrix)


    # save metrics and data

    # config eval
    # do_train = False

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









