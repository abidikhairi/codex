import glob
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    processors,
    normalizers,
    trainers
)
from transformers import PreTrainedTokenizerFast, AddedToken

INITIAL_VOCAB = [
    "{", "}", # Used for multiple instructions/input/output
    "[", "]", # One instruction delimiters
    "=", # Used to start a instruction, input, or output
    "<%", "%>", # Protein sequence delimiters
    "#_", # Amino acid placeholder used for sequence completion
    ".", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0",
    "EC", "MolecularFunction", "BiologicalProcess", "CellularComponent", 
    "Mutation", "SuperFamily",
    "Annotate", "Describe", "Generate", "Complete", # Instructions actions
    "instructions", "inputs", "outputs", '\n' # Prompt tokens
]

def load_protein_vocab():
    protein_tokenizer_file = 'data/protein_tokenizer.json'
    tokenizer = Tokenizer.from_file(protein_tokenizer_file)
    
    words = list(tokenizer.get_vocab().keys())

    words = list(map(lambda x: f"#{x}", words))
    return words


def main():
    protein_words = load_protein_vocab()
    print(f'Loaded protein words: {len(protein_words)} word')
    
    INITIAL_VOCAB_TOKENS = list(map(lambda x: AddedToken(x, single_word=True, lstrip=True, rstrip=True), INITIAL_VOCAB)) \
        + list(map(lambda x: AddedToken(x, single_word=True, rstrip=True, lstrip=True), protein_words))
    
    vocab_size = 15_000
    input_folder = "data/text_data"
    input_files = glob.glob(f'{input_folder}/*.txt')
    
    # Init special tokens
    unk_token="<|endoftext|>"
    bos_token="<|startoftext|>"
    eos_token="<|endoftext|>"
    pad_token="<|endoftext|>"
    
    special_tokens = [bos_token, eos_token, unk_token, pad_token]
    
    actual_vocab_size = vocab_size + len(special_tokens) + len(INITIAL_VOCAB)
    
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.add_tokens(INITIAL_VOCAB_TOKENS)
    
    print("Number of initial tokens: " + str(len(INITIAL_VOCAB_TOKENS)))
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase(), 
        normalizers.Strip(),
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.CharDelimiterSplit('\n')
    ])
    
    trainer = trainers.BpeTrainer(
                                  vocab_size=actual_vocab_size,
                                  special_tokens=special_tokens,
                                  initial_alphabet=INITIAL_VOCAB,
                                  show_progress=True)
    
    print(input_files)
    tokenizer.train(input_files, trainer=trainer)
    
    print(f'Number of tokens after training: %d' % tokenizer.get_vocab_size())
    bos_token_id = tokenizer.token_to_id(bos_token)
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos_token} $A",
        special_tokens=[(bos_token, bos_token_id)]
    )
        
    tokenizer_wrapper = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=bos_token,
                            eos_token=eos_token,
                            pad_token=pad_token,
                            unk_token=unk_token,
                            clean_up_tokenization_spaces=True,
                            split_special_tokens=True)
    
    tokenizer_wrapper.save_pretrained('data/tokenizer')

if __name__ == '__main__':
    main()
