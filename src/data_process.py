import json
import gzip

# filter some noises caused by speech recognition
# This is the function from the original paper of the dataset
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
        processed_sample['topic_list'] = sample['topic_list']
        processed_sample['general_query_list'] = sample['general_query_list']
        joined_transcript = '\n'.join(
            ["Speaker: " + i["speaker"] + "\n" + "Content: " + i["content"] for i in sample['meeting_transcripts']])
        processed_sample["meeting_transcripts"] = clean_data(joined_transcript)
        processed_data.append(processed_sample)
    
    # Print one sample to test
    print("\n--------------------")
    print("✅ Successfully preprocessed data!")
    print("--------------------")
    print(processed_data[0]['general_query_list'])
    print(processed_data[0]['meeting_transcripts'])
    print()

    return processed_data


def get_processed_data(dataset, train=True):
    if dataset == "qmsum":
        data = process_qmsum(train=train)
    # Insert more dataset if needed
    else:
        raise Exception("Dataset Error")

    return data


def test_data_generation(dataset, sample):
    res = []
    if dataset == 'qmsum':
        all_topic = ', '.join(item['topic'] for item in sample['topic_list'])
        for test_query in sample['general_query_list']:
            data = {}
            data['rag_query'] = test_query['query'] + ' The topic list of the meeting transcript is: ' + all_topic
            data['query'] = test_query['query']
            data['summary'] = test_query['answer']
            res.append(data)
    else:
        raise Exception("Dataset Error")
    return res


def split_corpus_into_chunk(dataset, sample, text_splitter):
    chunk_list = []
    if dataset == "qmsum":
        doc_list = [sample["meeting_transcripts"]]
    else:
        raise Exception("Dataset Error")

    for doc in doc_list:
        chunk_list.extend(text_splitter.split_text(doc))

    return chunk_list
    


