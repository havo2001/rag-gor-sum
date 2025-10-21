import json
import gzip

# filter some noises caused by speech recognition
# This is the function from the original paper
def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text

# We use the exactly preprocess procedure from the original paper
def process_qmsum(train=True):
    data_path = './data/raw/QMSum/ALL/jsonl/'
    data_path += 'train.jsonl' if train else 'test.jsonl' 
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    processed_data = []
    for sample in data:
        processed_sample = {}
        # I checked the training, all of them have one general query list: 'Summarize the whole meeting.'
        # For testing, only one has two answers for the general query list
        processed_sample['general_query_list'] = sample['general_query_list']
        joined_transcript = '\n'.join(
            ["Speaker: " + i["speaker"] + "\n" + "Content: " + i["content"] for i in sample['meeting_transcripts']])
        processed_sample["meeting_transcripts"] = clean_data(joined_transcript)
        processed_data.append(processed_sample)
    
    # Print one sample to test
    print(processed_data[0]['general_query_list'])
    print(processed_data[0]['meeting_transcripts'])
    print()

    # Save to data/processed/QMSum
    output_path = './data/processed/QMSum/processed_'
    output_path +=  'train.jsonl.gz' if train else 'test.jsonl.gz' 
    with gzip.open(output_path, 'wt', encoding='utf-8') as g:
        for sample in processed_data:
            g.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print('Successfully preprocess and save the processed data!')

    return processed_data
    

def main():
    train_qmsum = process_qmsum(train=True)
    test_qmsum = process_qmsum(train=False)


if __name__ == '__main__':
    main()

    


