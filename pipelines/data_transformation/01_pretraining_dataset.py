import pandas as pd
from tokenizers import Tokenizer

def tokenizer_sequence(sequence, tokenizer: Tokenizer):
    tokens = tokenizer.encode(sequence).tokens

    tokens = list(map(lambda x: f'#{x}', tokens))
    tokens = " ".join(tokens)
    
    return tokens


def main():
    input_file = "data/uniprot_sprot.csv"
    protein_tokenizer_path = "data/protein_tokenizer.json"
    
    tokenizer = Tokenizer.from_file(protein_tokenizer_path)
    
    uniprot_df = pd.read_csv(input_file)
    
    uniprot_df['Sequence'] = uniprot_df['Sequence'].map(lambda x: tokenizer_sequence(x, tokenizer))    
    
    num_samples = len(uniprot_df)
    num_train_samples = num_samples - 10000
    
    print('Split dataset into train/valid/test')    
    train_df = uniprot_df.sample(n=num_train_samples)
    test_df = uniprot_df.drop(train_df.index)
    
    valid_df = train_df.sample(n=2000)
    train_df = train_df.drop(valid_df.index)

    print('Saving splits sequences')
    train_df[['Sequence']].to_csv('data/pretraining/train_sequences.txt', header=False, index=False)
    valid_df[['Sequence']].to_csv('data/pretraining/valid_sequences.txt', header=False, index=False)
    test_df[['Sequence']].to_csv('data/pretraining/test_sequences.txt', header=False, index=False)

    print('Saving splits sequences ids')
    train_df[['Entry']].to_csv('data/train_ids.csv', index=False)
    valid_df[['Entry']].to_csv('data/valid_ids.csv', index=False)
    test_df[['Entry']].to_csv('data/test_ids.csv', index=False)


if __name__ == '__main__':
    main()
