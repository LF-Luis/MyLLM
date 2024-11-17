import os
import math
import multiprocessing
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import tiktoken
from src.utils.root import create_temp_data_dir
import time

'''
Data downloader file to download and process large datasets from HF.
This code can download (in parallel), tokenize data (in parallel) and store that data 
into multiple shards for usage later.

Right now this code is only being used for the "HuggingFaceFW/fineweb-edu" "sample-10BT"
dataset (~85GB), but it can be expanded as needed.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
'''

DEBUG_STREAMING = False # Here streaming is used just for debugging, though it's more powerful than just for that.
DEBUG_ENTRY_COUNT = 0
NUM_PROC = NUM_PROC_FOR_DOWNLOAD = int(0.75 * os.cpu_count())
HF_DATA_FIELD = "text"
HF_DATA_PATH = "HuggingFaceFW/fineweb-edu"
HF_DATA_SUBSET_NAME = "sample-10BT"
HF_DATA_SPLIT = "train"
SHARD_STORAGE_DIR_NAME = "edu_fineweb10B"
SHARD_FILE_PREFIX = "edufineweb"
CHUNK_SIZE = 16
SHARDS_COUNT = 100

if os.getenv('DEBUG_MODE'):
    # Debug args -- downloads only 200 entries from FineWeb EDU
    DEBUG_STREAMING = True
    NUM_PROC_FOR_DOWNLOAD = None # If streaming, can't use parallel downloading of dataset
    DEBUG_ENTRY_COUNT = 200
    SHARDS_COUNT = 7


# Using fast BPE tokenizer tiktoken 
tokenizer = tiktoken.get_encoding("r50k_base")
eot_token = tokenizer._special_tokens['<|endoftext|>']
def tokenize_to_uint16(data_entry):
    # Tokenize
    text = data_entry[HF_DATA_FIELD]
    tokens = [eot_token]
    tokens.extend(tokenizer.encode_ordinary(text))
    tokens = np.array(tokens)
    # Convert tokens to np.uint16 and return
    assert (
        (0 <= tokens).all() and (tokens < 2**16).all()
    ), f'Error: Token values out of expected bounds. text: {text}. tokens: {tokens}'
    return tokens.astype(np.uint16)


if __name__ == '__main__':
    # Env set needed due to py-3.8 and mac-os issues
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    start_time = time.time()

    shard_storage_dir = create_temp_data_dir(SHARD_STORAGE_DIR_NAME)

    # Load the dataset
    print(f'Storing shard data in: {shard_storage_dir}. Cached in: ~/.cache/huggingface/datasets/')
    dataset = load_dataset(
        HF_DATA_PATH, 
        name=HF_DATA_SUBSET_NAME, 
        split=HF_DATA_SPLIT,
        streaming=DEBUG_STREAMING,
        num_proc=NUM_PROC_FOR_DOWNLOAD,
    )

    if DEBUG_STREAMING:
        # Convert a streamed subset of the data into a Dataset obj, which is expected in the code below
        dataset_small_iterable = dataset.take(DEBUG_ENTRY_COUNT)
        examples = list(dataset_small_iterable)
        dataset = Dataset.from_list(examples)

    # Shuffle dataset before processing
    dataset = dataset.shuffle(seed=42)
    
    dataset_count = len(dataset)
    dataset_per_shard = math.ceil(dataset_count / SHARDS_COUNT)
    print(f'NUM_PROC: {NUM_PROC}')

    # Store every dataset_per_shard dataset group into a shard
    with multiprocessing.Pool(processes=NUM_PROC) as pool:
        token_sets = []
        shard_idx = 0
        datasets_processed = 0
        with tqdm(total=dataset_count, desc="Processing dataset", unit="entry") as pbar:

            for chunk_results in pool.imap(tokenize_to_uint16, dataset, chunksize=CHUNK_SIZE):
                token_sets.append(chunk_results)
                datasets_processed += 1
                pbar.update(1)

                if len(token_sets) >= dataset_per_shard or datasets_processed == dataset_count:
                    shard_tokens = np.concatenate(token_sets[:dataset_per_shard])
                    token_sets = token_sets[dataset_per_shard:]
                    file_path = os.path.join(
                        shard_storage_dir,
                        f"{SHARD_FILE_PREFIX}_i_{shard_idx:06d}_t_{len(shard_tokens)}.npy"
                    )
                    np.save(file_path, shard_tokens)
                    shard_idx += 1

        print(f'datasets_processed: {datasets_processed}')

    elapsed_time_secs = time.time() - start_time
    elapsed_time_mins = elapsed_time_secs / 60
    print(f"Time to download and process {SHARD_FILE_PREFIX} dataset: {elapsed_time_secs:.2f} seconds")
    print(f"({elapsed_time_mins:.2f} minutes)")

    ''' On mac, with 200 entries only and ending with 5 shards:
    Time to download edufineweb dataset: 20.03 seconds
    Or time to download: 0.33 minutes
    '''
    ''' On gpu_8x_a100_80gb_sxm4, full dataset
    Time to download and process edufineweb dataset: 2221.01 seconds
    (37.02 minutes).
    Each shard has around 99,700,000 tokens
    '''
