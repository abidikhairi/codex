import pandas as pd



def load_ids():
    train_ids = pd.read_csv('data/train_ids.csv')['Entry'].tolist()
    valid_ids = pd.read_csv('data/valid_ids.csv')['Entry'].tolist()
    test_ids = pd.read_csv('data/test_ids.csv')['Entry'].tolist()
    
    return train_ids, valid_ids, test_ids

def prepare_sequence(sequence: str):
    sequence = " ".join(map(lambda x: f'#{x}', list(sequence)))
    return f'<% {sequence} %>'

def prepare_family_name(family_name: str):
    return f'[{family_name}]'

def merge_datasets(sequence_df: pd.DataFrame, family_df: pd.DataFrame):
    generation_df = sequence_df.merge(family_df, left_on='Entry', right_on='Entry', how='left').dropna()
    
    generation_df['Sequence'] = generation_df['Sequence'].apply(prepare_sequence)
    generation_df['Family'] = generation_df['Family'].map(prepare_family_name)

    return generation_df

def get_generation_dataset(merged_data: pd.DataFrame):
    merged_data['Instructions'] = "instructions={[Generate By Super Family]}"
    merged_data['Inputs'] = merged_data['Family'].map(lambda x: f'inputs={x}')
    merged_data['Outputs'] = merged_data['Sequence'].map(lambda x: f'outputs={x}')
    
    merged_data = merged_data[['Entry', 'Instructions', 'Inputs', 'Outputs']]
    
    return merged_data

def get_annotation_dataset(merged_data: pd.DataFrame):
    merged_data['Instructions'] = "instructions={[Annotate By Super Family]}"
    merged_data['Inputs'] = merged_data['Sequence'].map(lambda x: f'inputs={x}')
    merged_data['Outputs'] = merged_data['Family'].map(lambda x: f'outputs={x}')
    
    merged_data = merged_data[['Entry', 'Instructions', 'Inputs', 'Outputs']]
    
    return merged_data


def main():
    input_file = "data/uniprot_super_families.csv"
    sequence_file = "data/uniprot_sprot.csv"
    
    family_df = pd.read_csv(input_file)
    sequence_df = pd.read_csv(sequence_file)
    
    train_ids, valid_ids, test_ids = load_ids()

    merged_data = merge_datasets(sequence_df=sequence_df, family_df=family_df)
    
    generation_data = get_generation_dataset(merged_data)
    annotation_data = get_annotation_dataset(merged_data)
    
    concatenated_data = pd.concat([generation_data, annotation_data])

    train_data = concatenated_data[concatenated_data['Entry'].isin(train_ids)]
    valid_data = concatenated_data[concatenated_data['Entry'].isin(valid_ids)]
    test_data = concatenated_data[concatenated_data['Entry'].isin(test_ids)]
    
    train_data.to_csv('data/instructions/super_families/train.csv', index=False)
    valid_data.to_csv('data/instructions/super_families/valid.csv', index=False)
    test_data.to_csv('data/instructions/super_families/test.csv', index=False)

if __name__ == '__main__':
    main()
