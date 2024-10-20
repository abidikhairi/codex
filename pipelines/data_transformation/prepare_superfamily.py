import pandas as pd


def main():
    input_file = 'data/downloaded/Uniprot_GO_AND_INTERPRO.tsv'
    family_input_file = "data/InterPro_Superfamilies.csv"
    
    family_df = pd.read_csv(family_input_file, sep=';', names=['InterPro', "Family"])


    uniprot_df = pd.read_csv(input_file, sep='\t')
    uniprot_df = uniprot_df[['Entry', "InterPro"]]
    uniprot_df = uniprot_df.dropna()
    
    uniprot_df['InterPro'] = uniprot_df['InterPro'].map(lambda x: x.split(';'))

    uniprot_df = uniprot_df.explode('InterPro')
    uniprot_df = uniprot_df[uniprot_df['InterPro'].str.len() > 1]
    
    uniprot_df = uniprot_df.merge(family_df, left_on='InterPro', right_on='InterPro', how='left')
    uniprot_df = uniprot_df.dropna().drop(columns='InterPro')
        
    uniprot_df.to_csv('data/uniprot_super_families.csv', index=False)    

if __name__ == '__main__':
    main()
