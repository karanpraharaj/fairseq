import os

data = 'marc'

if data == 'mtop':
    MODEL = 'xmod.base'
    languages = ['de', 'en', 'fr', 'th', 'es', 'hi']
    partitions = range(5)

    for language in languages:
        for partition in partitions:
            print(language, partition)
            train_path = f"mtop_partitioned/train/{language}/{partition}.jsonl"
            valid_path = f"mtop_partitioned/eval/{language}/eval.jsonl"
            # test_path = f"mtop_partitioned/test/{language}/test.jsonl"
            dest_dir = f"mtop_partitioned/fairseq/{language}/{partition}"
            os.system(f"python preprocess_mtop.py \
                --sentencepiece-model {MODEL}/sentencepiece.bpe.model \
                --train {train_path} \
                --valid {valid_path} \
                --destdir {dest_dir}")

elif data == 'marc':
    MODEL = 'xmod.base'
    languages = ['de', 'en', 'fr', 'jp', 'zh']
    categories = ['apparel', 'home', 'musical_instruments', 'sports']

    # for language in languages:
    #     for category in categories:
    #         print(language, category)
    #         train_path = f"marc/train/{language}/{category}/train.jsonl"
    #         valid_path = f"marc/eval/{language}/eval.jsonl"
    #         # test_path = f"mtop_partitioned/test/{language}/test.jsonl"
    #         dest_dir = f"marc/fairseq/{language}/{category}"
            
    #         os.system(f"python ./marc/preprocess_marc.py \
    #             --sentencepiece-model {MODEL}/sentencepiece.bpe.model \
    #             --train {train_path} \
    #             --valid {valid_path} \
    #             --destdir {dest_dir}")


    # Preprocess IPT

    for language in languages:
            print(language, 'IPT')
            train_path = f"marc/train/IPT/{language}/train.jsonl"
            valid_path = f"marc/eval/{language}/eval.jsonl"
            # test_path = f"mtop_partitioned/test/{language}/test.jsonl"
            dest_dir = f"marc/fairseq/IPT/{language}"
            
            os.system(f"python ./marc/preprocess_marc.py \
                --sentencepiece-model {MODEL}/sentencepiece.bpe.model \
                --train {train_path} \
                --valid {valid_path} \
                --destdir {dest_dir}")