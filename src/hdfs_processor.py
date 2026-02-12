import re
import pandas as pd
from drain_parser import DrainParser
import pickle

def extract_block_id(log_text):
    match = re.search(r'(blk_-?\d+)', log_text)
    return match.group(1) if match else None

def process_hdfs_logs(file_path):
    parser = DrainParser()
    data = []
    
    print(f"Opening {file_path}... This may take a moment.")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            bid = extract_block_id(line)
            if bid:
                parsed_row = parser.parse([line])
                data.append({
                    "BlockID": bid,
                    "EventID": parsed_row.iloc[0]['EventID']
                })
            
            # Progress tracker every 500k lines
            if i % 500000 == 0 and i > 0:
                print(f"Parsed {i} lines...")

    df = pd.DataFrame(data)
    print("Grouping sequences by BlockID...")
    hfdf_sequences = df.groupby('BlockID')['EventID'].apply(list).reset_index()
    return hfdf_sequences

if __name__ == "__main__":
    seq_df = process_hdfs_logs('HDFS.log')
    print(f"Total Block Sequences: {len(seq_df)}")
    # Save processed data to save time in future runs
    seq_df.to_pickle('hdfs_sequences.pkl')
    print("Saved sequences to 'hdfs_sequences.pkl'")
