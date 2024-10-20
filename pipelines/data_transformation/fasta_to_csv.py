import pandas as pd
from Bio import SeqIO

def main():
    input_file = "data/downloaded/uniprot_sprot.fasta"

    data = {
        "Entry": [],
        "Sequence": []
    }
    
    for record in SeqIO.parse(input_file, format='fasta'):
        seq = record.seq
        entry = record.id.split('|')[1]
        
        if len(seq) > 512:
            continue
        
        data["Entry"].append(entry)
        data["Sequence"].append(str(seq))
        
        
    pd.DataFrame(data).to_csv('data/uniprot_sprot.csv', index=False)

if __name__ == '__main__':
    main()
