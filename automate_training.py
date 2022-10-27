'''
Automate the training for swin classifier
'''


from ai_helper import constants_training as c_t
from ai_helper import constants_huggingface as c_h
from helper import erik_functions_files as e_f

import classes_eurosat
import time




if __name__=='__main__':
    #opt_lr = [4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5]
    #opt_lr = [2.5e-4, 1.5e-4, 1e-4, 5e-5, 2.5e-5]
    #opt_lr = [2.5e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 3e-5]
    opt_lr = [1e-4]

    path_swinv2_config = r'C:\Users\erikw\git\eurosat\config_swinv2_base.json'
    path_dataset_config = r'C:\Users\erikw\git\eurosat\config_dataset_base.json'

    for lr in opt_lr:
        config_swinv2 = e_f.json_load(path_swinv2_config)
        config_dataset = e_f.json_load(path_dataset_config)

        config_dataset[c_t.DATASET_SMALL] = c_t.IS_TRUE
        config_dataset[c_t.DATASET_SMALL] = c_t.IS_FALSE

        config_swinv2[c_t.S2_TRAIN_ARGS_LR] = lr * 1.01
        #config_swinv2[c_t.S2_MODEL_CHECKPOINT] = c_h.MODEL_VISION_SWINV2_MICROSOFT_SMALL_PATCH4_WINDOW8_256
        config_swinv2[c_t.S2_MODEL_CHECKPOINT] = c_h.MODEL_VISION_SWINV2_MICROSOFT_TINY_PATCH4_WINDOW8_256
        config_swinv2[c_t.S2_TRAIN_ARGS_GRADIENT_ACCUMULATION_STEPS] = 1
        config_swinv2[c_t.S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE] = 64
        config_swinv2[c_t.S2_TRAIN_ARGS_PER_DEVICE_EVAL_BATCH_SIZE] = 64
        config_swinv2[c_t.S2_TRAIN_ARGS_NUM_TRAIN_EPOCHS] = 5

        print('lr : ' + str(config_swinv2[c_t.S2_TRAIN_ARGS_LR]))
        print('epochs : ' + str(config_swinv2[c_t.S2_TRAIN_ARGS_NUM_TRAIN_EPOCHS]))
        print('dataset_small : ' + str(config_dataset[c_t.DATASET_SMALL]))
        print('checkpoint model : ' + str(config_swinv2[c_t.S2_MODEL_CHECKPOINT]))
        print('batch size : ' + str(config_swinv2[c_t.S2_TRAIN_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE]))

        print()
        print(config_swinv2)
        print(config_dataset)
        print()

        # create the trainer for a swinv2 model
        swin_trainer = classes_eurosat.SwinTraining(config_json=config_swinv2, config_json_dataset=config_dataset)

        time.sleep(5)
        swin_trainer.train()
        time.sleep(5)
        swin_trainer.data_save()



