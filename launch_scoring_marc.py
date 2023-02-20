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
    # test_df = pd.read_csv(f"marc/test/{language}/test.tsv", sep='\t')
    categories = ['apparel', 'home', 'musical_instruments', 'sports']

    # Concatenate all test files of a language and return a dataframe
    test_df = pd.DataFrame()

    for category in categories:
        print('====LANGUAGE: ', language, '====')
        print('Concatenating test files of category: ', category)
        test_df = pd.concat([test_df, pd.read_csv(f"marc/test/{language}/{category}/test.tsv", sep='\t')])
        
    print('Number of test examples: ', len(test_df))
    print('Language of test data: ', language)
    return test_df

def predict_test_df(df, adapter_lang, model):
    print('Adapter used for prediction: ', adapter_lang)
    start_time = time.time()
    df['model'] = [model] * len(df)
    df['adapter_lang'] = adapter_lang * len(df)
    df['predicted_label'] = df['text'].swifter.apply(lambda x: predict(x, adapter_lang, model))
    end_time = time.time()
    time_taken = end_time - start_time
    
    print('Time taken to predict: ', int(time_taken/60), ' minutes and ', round(time_taken%60,2), ' seconds')

    return df

def calculate_f1(df):
    # Convert df['predicted_label'] to int
    df['predicted_label'] = df['predicted_label'].astype(int)
    return round(f1_score(df['label'], df['predicted_label'], average='weighted'), 4)

def calculate_accuracy(df):
    # Convert df['predicted_label'] to int
    df['predicted_label'] = df['predicted_label'].astype(int)
    return round(accuracy_score(df['label'], df['predicted_label']), 4)

def compute(test_lang, adapter_lang, model):
    test_df = prepare_test_df(test_lang)
    test_df = predict_test_df(test_df, adapter_lang, model)
    f1 = calculate_f1(test_df) * 100
    accuracy = calculate_accuracy(test_df) * 100
    return round(f1,2), round(accuracy,2)


lang_adapter_map = {'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'fr': 'fr_XX', 'zh': 'zh_CN', 'jp': 'ja_XX'}

seq_num = input('Enter sequence number: ')

# Populating checkpoint subdirectories
chkpt_dir = f'checkpoints/marc/seq_{seq_num}/'

# List all subdirectories using os.listdir [SEQ 0]
chkpt_subdirectories = [
            #  "hop_IPT_de", "hop_IPT_en", "hop_IPT_jp", "hop_IPT_zh", "hop_IPT_fr",
            #  "hop_1_zh_home", "hop_2_de_sports", "hop_3_en_sports", "hop_4_fr_apparel", 
            #  "hop_5_fr_musical_instruments",
            #  "hop_6_en_apparel", "hop_7_de_musical_instruments", "hop_8_en_musical_instruments", "hop_9_fr_home", "hop_10_jp_home",
            #  "hop_11_zh_musical_instruments", "hop_12_zh_apparel", "hop_13_jp_sports", "hop_14_de_apparel", "hop_15_jp_musical_instruments",
            #  "hop_16_de_home", "hop_17_fr_sports", "hop_18_zh_sports", "hop_19_jp_apparel", "hop_20_en_home",
            # "hop_1_zh_home_low_lr", 
            # "hop_2_de_sports_low_lr", "hop_3_en_sports_low_lr", "hop_4_fr_apparel_low_lr", "hop_5_fr_musical_instruments_low_lr",
            #  "hop_6_en_apparel_low_lr", "hop_7_de_musical_instruments_low_lr",
             "hop_8_en_musical_instruments_low_lr", "hop_9_fr_home_low_lr", "hop_10_jp_home_low_lr", "hop_11_zh_musical_instruments_low_lr",
             "hop_12_zh_apparel_low_lr", "hop_13_jp_sports_low_lr", "hop_14_de_apparel_low_lr", "hop_15_jp_musical_instruments_low_lr",
             "hop_16_de_home_low_lr", "hop_17_fr_sports_low_lr", "hop_18_zh_sports_low_lr", "hop_19_jp_apparel_low_lr", "hop_20_en_home_low_lr",
]           

# List all subdirectories using os.listdir [SEQ 2]

# chkpt_subdirectories = [
#     "hop_IPT_de",
#     "hop_IPT_en",
#     "hop_IPT_fr",
#     "hop_IPT_jp",
#     "hop_IPT_zh",
#     "hop_1_fr_musical_instruments",
#     "hop_2_zh_apparel",
#     "hop_3_en_home",
#     "hop_4_jp_apparel",
#     "hop_5_en_sports",
#     "hop_6_jp_home",
#     "hop_7_zh_sports",
#     "hop_8_fr_home",
#     "hop_9_de_musical_instruments",
#     "hop_10_jp_sports",
#     "hop_11_zh_musical_instruments",
#     "hop_12_en_apparel",
#     "hop_13_en_musical_instruments",
#     "hop_14_de_apparel",
#     "hop_15_zh_home",
#     "hop_16_fr_apparel",
#     "hop_17_jp_musical_instruments",
#     "hop_18_de_sports",
#     "hop_19_de_home",
#     "hop_20_fr_sports",
# ]


for chkpt in chkpt_subdirectories:
    MODEL=f'checkpoints/marc/seq_{seq_num}/{chkpt}'
    hop_train_language = chkpt.split('_')[2]
    
    if chkpt.split('_')[1] == 'IPT':
        hop_train_partition = hop_train_language
        hop_train_language = 'IPT'
        
    else:
        hop_train_partition = chkpt.split('_')[3]
        if hop_train_partition == 'musical':
            hop_train_partition = 'musical_instruments'

    DATA = f'marc/fairseq/{hop_train_language}/{hop_train_partition}/bin' # Training data

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
    results_subdir = f'results/marc/seq_{seq_num}/test/{chkpt}'
    
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    chkpt_results_df = pd.DataFrame(columns=['test_lang', 'adapter_lang', 'f1', 'accuracy'])

    for test_lang in ['de', 'en', 'fr', 'jp', 'zh']:
        test_lang_df = pd.DataFrame(columns=['test_lang', 'adapter_lang', 'f1', 'accuracy'])
        for adapter_lang in ['de', 'en', 'fr', 'jp', 'zh']:
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
    