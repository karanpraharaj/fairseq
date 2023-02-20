import pandas as pd
from fairseq.models.xmod import XMODModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import os
import numpy as np
import swifter

def predict(sentence, lang, model):
    adapter_lang = lang_adapter_map[lang]
    tokens = model.encode(sentence)
    idx = model.predict('sentence_classification_head', tokens, lang_id=[adapter_lang]).argmax().item()
    dictionary = model.task.label_dictionary
    return dictionary[idx + dictionary.nspecial]

def prepare_test_df(language):
    test_df = pd.read_csv(f"mtop_partitioned/test/{language}/test.tsv", sep='\t')
    print('Number of test examples: ', len(test_df))
    print('Language of test data: ', language)
    return test_df

def predict_test_df(df, adapter_lang, model):
    print('Adapter used for prediction: ', adapter_lang)
    start_time = time.time()
    df['model'] = [model] * len(df)
    df['adapter_lang'] = adapter_lang * len(df)
    df['predicted_intent'] = df['text'].swifter.apply(lambda x: predict(x, adapter_lang, model))
    end_time = time.time()
    time_taken = end_time - start_time
    
    print('Time taken to predict: ', int(time_taken/60), ' minutes and ', round(time_taken%60,2), ' seconds')

    return df

def calculate_f1(df):
    return round(f1_score(df['intent'], df['predicted_intent'], average='weighted'), 4)

def calculate_accuracy(df):
    return round(accuracy_score(df['intent'], df['predicted_intent']), 4)

def compute(test_lang, adapter_lang, model):
    test_df = prepare_test_df(test_lang)
    test_df = predict_test_df(test_df, adapter_lang, model)
    f1 = calculate_f1(test_df) * 100
    accuracy = calculate_accuracy(test_df) * 100
    return round(f1,2), round(accuracy,2)


lang_adapter_map = {'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'fr': 'fr_XX', 'th': 'th_TH', 'hi': 'hi_IN'}

seq_num = input('Enter sequence number: ')

# Populating checkpoint subdirectories
chkpt_dir = f'checkpoints/seq_{seq_num}/'

# List all subdirectories using os.listdir [SEQ 0]
# chkpt_subdirectories = ['hop_IPT_de_0', 'hop_IPT_fr_0','hop_IPT_hi_0', 'hop_IPT_es_0', 'hop_IPT_en_0',
#                         'hop_IPT_th_0', 'hop_2_fr_1', 'hop_3_hi_1', 'hop_4_es_1', 'hop_5_th_1', 
#                         'hop_6_es_2', 'hop_7_en_1', 'hop_8_de_1', 'hop_9_en_2',  'hop_10_de_2',
#                         'hop_11_fr_2', 'hop_12_hi_2', 'hop_13_es_3', 'hop_14_hi_3', 'hop_15_en_3', 
#                         'hop_16_th_2', 'hop_17_fr_3', 'hop_18_hi_4', 'hop_19_de_3', 'hop_20_es_4', 
#                         'hop_21_th_3', 'hop_22_fr_4', 'hop_23_de_4', 'hop_24_th_4', 'hop_25_en_4',  
#                         ]

chkpt_subdirectories = ['hop_18_hi_4', 'hop_19_de_3', 
                        'hop_21_th_3', 'hop_22_fr_4', 'hop_23_de_4', 'hop_24_th_4', 'hop_25_en_4',  
                        ]


# List all subdirectories using os.listdir [SEQ 1]
# chkpt_subdirectories = [
#     'hop_IPT_th_0', 'hop_IPT_en_0', 'hop_IPT_es_0', 'hop_IPT_hi_0', 'hop_IPT_de_0',
#     'hop_IPT_fr_0', 'hop_2_th_1',   'hop_3_hi_1',   'hop_4_fr_1',   'hop_5_de_1',   
#     'hop_6_es_1',   'hop_7_en_1',   'hop_8_es_2',   'hop_9_hi_2',   'hop_10_th_2',
#     'hop_11_de_2',  'hop_12_fr_2',  'hop_13_es_3',  'hop_14_en_2',  'hop_15_de_3',
#     'hop_16_th_3',  'hop_17_en_3',  'hop_18_fr_3',  'hop_19_hi_3',  'hop_20_hi_4',
#     'hop_21_es_4',  'hop_22_fr_4',  'hop_23_th_4',  'hop_24_de_4',  'hop_25_en_4',
#     ]

# chkpt_subdirectories = [
#                         'hop_IPT_de_0',
#                         # 'hop_5_de_1', 
#                         # 'hop_9_hi_2', 
#                         'hop_13_es_3', 
#                         'hop_17_en_3',
#                         'hop_21_es_4',
#                         'hop_25_en_4',  
#                         ]

# List all subdirectories using os.listdir [SEQ 2]
# chkpt_subdirectories = ['hop_IPT_de_0', 'hop_IPT_fr_0','hop_IPT_hi_0', 'hop_IPT_en_0',
#                         'hop_2_fr_1',   'hop_3_hi_1',  'hop_4_en_1',   'hop_5_de_1', 
#                         'hop_6_en_2',  'hop_7_de_2',   'hop_8_fr_2',   'hop_9_hi_2', 
#                         'hop_10_hi_3', 'hop_11_en_3',  'hop_12_fr_3',  'hop_13_hi_4', 
#                         'hop_14_de_3', 'hop_15_fr_4',  'hop_16_de_4',  'hop_17_en_4',  
#                         ]

# List all subdirectories using os.listdir [SEQ 3]
chkpt_subdirectories = [
                        'hop_IPT_de_0', 'hop_IPT_th_0', 'hop_IPT_en_0', 'hop_IPT_fr_0', 'hop_IPT_hi_0', 'hop_IPT_es_0', 
                         'hop_2_en_1', 'hop_3_fr_1', 'hop_4_th_1', 'hop_5_hi_1', 'hop_6_de_1', 'hop_7_es_1', 
                         'hop_8_fr_2', 'hop_9_hi_2', 'hop_10_en_2', 
                         'hop_11_th_2', 'hop_12_es_2', 'hop_13_de_2', 
                         'hop_14_fr_3', 'hop_15_hi_3', 'hop_16_th_3', 'hop_17_en_3', 'hop_18_de_3', 'hop_19_es_3', 
                         'hop_20_es_4', 'hop_21_de_4', 'hop_22_fr_4', 'hop_23_en_4', 'hop_24_hi_4', 'hop_25_th_4'
                         ]

for chkpt in chkpt_subdirectories:
    MODEL=f'checkpoints/seq_{seq_num}/{chkpt}'
    hop_train_language = chkpt.split('_')[2]
    hop_train_partition = chkpt.split('_')[3]
    DATA = f'mtop_partitioned/fairseq/{hop_train_language}/{hop_train_partition}/bin' # Training data

    # Load model
    model = XMODModel.from_pretrained(
                model_name_or_path=MODEL,
                checkpoint_file='checkpoint_best.pt', 
                data_name_or_path=DATA, 
                suffix='', 
                criterion='cross_entropy', 
                bpe='sentencepiece',  
                sentencepiece_model=DATA+'/input0/sentencepiece.bpe.model')
    
    # results_subdir = f'results/seq_{seq_num}/test_all_combinations/{chkpt}'
    results_subdir = f'results/seq_{seq_num}/test/{chkpt}'
    
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    chkpt_results_df = pd.DataFrame(columns=['test_lang', 'adapter_lang', 'f1', 'accuracy'])

    for test_lang in ['de', 'en', 'fr', 'th', 'es', 'hi']:
        test_lang_df = pd.DataFrame(columns=['test_lang', 'adapter_lang', 'f1', 'accuracy'])
        for adapter_lang in ['de', 'en', 'fr', 'th', 'es', 'hi']:
            if test_lang == adapter_lang:
                f1, accuracy = compute(test_lang, adapter_lang, model)
                print("Training hop details:")
                print("checkpoint: ", chkpt)
                print("test_lang: ", test_lang)
                print("adapter_lang: ", adapter_lang)
                print("\n")
                print("f1: ", f1)
                print("accuracy: ", accuracy)
                print("\n")
                print('--------------------------------------------------')

                chkpt_results_df = chkpt_results_df.append({'test_lang': test_lang, 'adapter_lang': adapter_lang, 'f1': f1, 'accuracy': accuracy}, ignore_index=True)
                test_lang_df = test_lang_df.append({'test_lang': test_lang, 'adapter_lang': adapter_lang, 'f1': f1, 'accuracy': accuracy}, ignore_index=True)
        print('==================================================')

        if not os.path.exists(f'{results_subdir}/{test_lang}'):
            os.makedirs(f'{results_subdir}/{test_lang}')
        
        test_lang_df.to_csv(f'{results_subdir}/{test_lang}/results.log', sep='\t', index=False)

    print('Saving results to logs...')

    # if os.path.exists(f'{results_subdir}/results.log'):
    #    chkpt_results_df.to_csv(f'{results_subdir}/results_all_combs.log', sep='\t', index=False)
    # else:
    chkpt_results_df.to_csv(f'{results_subdir}/results.log', sep='\t', index=False)
    