from tokenizers import Tokenizer

def load_protein_tokenizer():
    input_file = "data/protein_tokenizer.json"
    tokenizer = Tokenizer.from_file(input_file)
    
    return tokenizer


def tokenize_protein(tokenizer, sequence):
    tokens = tokenizer.encode(sequence).tokens
    return " ".join(map(lambda x: f'#{x}', tokens))


def main():
    tokenizer = load_protein_tokenizer()
    
    sample_sequence = "MRWQEMGYIF"
    mutated_sequence = "MRWEEMGYIF"
    
    mutated_sequence_processed = tokenize_protein(tokenizer, mutated_sequence)
    processed_sequence = tokenize_protein(tokenizer, sample_sequence)
    
    annotation = ["vacuole inheritance", 'alpha-1,6-mannosyltransferase activity']

    instructions = 'instructions={[Annotate By Molecular Function]}'
    input = f"inputs=<% {processed_sequence} %>"
    output = "outputs={" + ",".join(map(lambda x: f'[{x}]', annotation)) + "}"
    
    annotation_prompt = """
        {instructions}
        {input}
        {output}
    """.format(instructions=instructions, input=input, output=output)

    print()
    print('Annotation Prompt:')
    print(annotation_prompt)
    
    instructions = 'instructions={[Generate By Superfamily]}'
    input = f"inputs=[Butyrophylin-like, SPRY domain]"
    output = f"outputs=<% {processed_sequence} %>"
    
    generation_prompt = """
        {instructions}
        {input}
        {output}
    """.format(instructions=instructions, input=input, output=output)
    
    print('Generation Prompt (By Superfamily name):')
    print(generation_prompt)

    instructions = 'instructions={[Describe Mutation]}'
    input = f"inputs=<% {processed_sequence} %> <% {mutated_sequence_processed} %>"
    output = f"outputs=[abolishes DNA-binding activity]"
    
    mutation_prompt = """
        {instructions}
        {input}
        {output}
    """.format(instructions=instructions, input=input, output=output)
    
    print('Mutation Prompt:')
    print(mutation_prompt)


    masked_sequence = "#MRW #_ #MG #_"
    masked_sequence_processed = " ".join(map(lambda x: f'#{x}', list(masked_sequence)))
    masked_sequence_processed = "#MRW #_ #MG #_"

    instructions = 'instructions={[Complete Sequence]}'
    input = f"inputs=<% {masked_sequence_processed} %>"
    output = f"outputs=<% {mutated_sequence_processed} %>"
    
    complete_seq_prompt = """
        {instructions}
        {input}
        {output}
    """.format(instructions=instructions, input=input, output=output)
    
    print('Complete Sequence:')
    print(complete_seq_prompt)
    
    instructions = 'instructions={[Generate By EC]}'
    input = f"inputs=[2.3.1.199]"
    output = f"outputs=<% {processed_sequence} %>"
    
    generation_prompt = """
        {instructions}
        {input}
        {output}
    """.format(instructions=instructions, input=input, output=output)
    
    print('Generation Prompt (By EC number):')
    print(generation_prompt)    


if __name__ == '__main__':
    main()
