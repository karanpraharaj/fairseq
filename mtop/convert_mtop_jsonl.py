import json



for language in ['de', 'en', 'es', 'fr', 'hi', 'th']:
    for split in ['train', 'eval', 'test']:
        data = []

        with open(f'/Users/karanpraharaj/fairseq/mtop/{language}/{split}.txt') as f:
            for i, line in enumerate(f):
                columns = line.strip().split('\t')
            
                example = {
                    'id': columns[0],
                    'intent': columns[1],
                    'slot_string': columns[2],
                    'utterance': columns[3],
                    'domain': columns[4],
                    'locale': columns[5],
                    'decoupled_form': columns[6],
                    'tokens_json': json.loads(columns[7])
                }
            
                data.append(example)

        with open(f'/Users/karanpraharaj/fairseq/mtop/{language}/{split}.jsonl', 'w') as f:
            for example in data:
                json.dump(example, f)
                f.write('\n')
        
        print(language, split, len(data))