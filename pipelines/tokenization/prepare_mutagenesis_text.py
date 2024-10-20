import re
import pandas as pd

def extract_position(text: str):
    text = text.split(';')[0]
    text = text.replace('MUTAGEN', '').strip()
    
    return text.strip()

def extract_effect(text: str):
    text = text.replace("\"", "").replace('.', '')
    text = text.split(';')[1]
    text = text.split(':')[1]
    
    return text.strip()

def extract_reference_aa(text: str):
    text = text.replace("\"", "").replace('.', '')
    text = text.split(';')[1]
    text = text.split(':')[0]
    text = text.replace("/note=", '')
    text = text.split('->')[0]
    return text.strip()

def extract_target_aa(text: str):
    text = text.replace("\"", "").replace('.', '')
    text = text.split(';')[1]
    text = text.split(':')[0]
    text = text.replace("/note=", '')
    try:
        text = text.split('->')[1]
    except IndexError:
        return None
    return text.strip()


def main():
    regex = r"MUTAGEN [0-9]+; \/note=\"[A-Za-z0-9->:\s,\/.]+\";"
    
    input_file = "data/downloaded/Uniprot_EC_AND_Mutagenesis.tsv"

    df = pd.read_csv(input_file, sep='\t')
    df = df.dropna()
     
    df['MutagenesisPattern'] = df['Mutagenesis'].map(lambda x: re.findall(regex, x))
    
    df = df.explode('MutagenesisPattern')
    df = df.dropna()
    
    df['Position'] = df['MutagenesisPattern'].map(extract_position)
    df['Effect'] = df['MutagenesisPattern'].map(extract_effect)
    df['RefAA'] = df['MutagenesisPattern'].map(extract_reference_aa)
    df['AltAA'] = df['MutagenesisPattern'].map(extract_target_aa)
    
    df = df[['Entry', 'Effect', 'Position', 'RefAA', 'AltAA']]
    df = df.dropna()

    df[['Effect']].to_csv('data/text_data/Uniprot_Mutagenesis.txt', header=None, index=False)
    df.to_csv('data/Uniprot_Mutagenesis.csv', index=False)

if __name__ == '__main__':
    main()
