import pandas as pd

def main():
    input_file = "data/downloaded/Uniprot_EC_AND_Mutagenesis.tsv"

    df = pd.read_csv(input_file, sep='\t')
    df = df.dropna()
     
    df['EC number'] = df['EC number'].map(lambda x: x.split(';'))
    df = df.explode('EC number')
    
    df = df[df['EC number'].str.contains('-') == False]
    df['EC number'] = df['EC number'].str.strip()
    
    
    df[['EC number']].to_csv('data/text_data/Uniprot_EC.txt', header=None, index=False)
    df[['Entry', 'EC number']].to_csv('data/Uniprot_EC.csv', index=False)

if __name__ == '__main__':
    main()
