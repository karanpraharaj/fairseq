import csv
import subprocess
import os

MODEL = "xmod.base"
MAX_EPOCH = 10
LR = "3e-05"
BATCH_SIZE = 8

restore_file = f"{MODEL}/model.pt"

adapter_map = {'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'fr': 'fr_XX', 'zh': 'zh_CN', 'jp': 'ja_XX'}

def read_csv(file_path):
    sequence_tuples = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sequence_tuples.append((row[0], row[1]))
    return sequence_tuples

seq_num = input('Enter sequence number: ')
csv_file = f'/Users/karanpraharaj/fairseq/sequence_csv/marc/seq_{seq_num}_IPT_expand.csv' # Hardcoded for now, but change later.
resume_from = 7

seq_tuples = read_csv(csv_file)

for i, hop_tuple in enumerate(seq_tuples):
    language, partition = hop_tuple
    DATA_DIR = f"marc/fairseq/{language}/{partition}/bin"
    
    if language == 'IPT':
        hop_num = 'IPT' 
    else:
        hop_num = i-4

    print('Starting hop #: ', hop_num)
    print('Language: ', language)
    print('Partition: ', partition)

    checkpoint_subdir = f'hop_{hop_num}_{language}_{partition}_low_lr'

    if language == 'IPT':
        language = partition
        checkpoint_subdir = f'hop_{hop_num}_{partition}'

    results_subdir = f'results/marc/seq_{seq_num}/low_lr/eval/{checkpoint_subdir}'

    # Create if results subdir doesn't exist
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir, exist_ok=True)

    cmd = f"fairseq-train {DATA_DIR} " \
            f"--restore-file {restore_file} " \
            f"--save-dir checkpoints/marc/seq_{seq_num}/{checkpoint_subdir} " \
            "--reset-optimizer " \
            "--reset-dataloader " \
            "--reset-meters " \
            "--best-checkpoint-metric accuracy " \
            "--maximize-best-checkpoint-metric " \
            "--task sentence_prediction_adapters " \
            "--num-classes 2 " \
            "--init-token 0 " \
            "--separator-token 2 " \
            "--max-positions 512 " \
            "--shorten-method 'truncate' " \
            "--arch xmod_base " \
            "--dropout 0.1 " \
            "--attention-dropout 0.1 " \
            "--weight-decay 0.01 " \
            "--criterion sentence_prediction_adapters " \
            "--optimizer adam " \
            "--adam-betas '(0.9, 0.98)' " \
            "--adam-eps 1e-06 " \
            "--clip-norm 0.0 " \
            "--lr-scheduler fixed " \
            f"--lr {LR} " \
            f"--batch-size {BATCH_SIZE} " \
            "--required-batch-size-multiple 1 " \
            "--update-freq 1 " \
            f"--max-epoch {MAX_EPOCH} " \
            f"--lang-id {adapter_map[language]} " \
            "--no-epoch-checkpoints " \
            "--no-last-checkpoints " \
            f"--log-file {results_subdir}/results.log " \
            "--log-format tqdm " \
    
    print('--------------------------------')
    print('Kicking off job: ', cmd)
    print('--------------------------------')

    if i > resume_from:
        subprocess.run(cmd, shell=True) # Runs the command and waits for it to complete.         
        #If we want to run the command in the background and continue executing the Python script, you can use subprocess.Popen() instead.
        print('Completed training hop #: ', hop_num, ' with: ', language, '-', partition)
    
    else:
        print('Skipping hop #: ', hop_num, ' with: ', language, '-', partition)

    restore_file = f'checkpoints/marc/seq_{seq_num}/{checkpoint_subdir}/checkpoint_best.pt'