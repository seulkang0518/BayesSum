from Bio import AlignIO

# --- Define your filenames ---
input_sto_file = 'PF12740.alignment.full'
output_fasta_file = 'PF12740.fasta'

print(f"Converting {input_sto_file} to FASTA format...")

try:
    # Read the Stockholm alignment file
    with open(input_sto_file, 'r') as f_in:
        # Write the sequences to a new file in FASTA format
        with open(output_fasta_file, 'w') as f_out:
            alignments = AlignIO.parse(f_in, 'stockholm')
            AlignIO.write(alignments, f_out, 'fasta')

    print(f"Success! File converted and saved as {output_fasta_file}")

except FileNotFoundError:
    print(f"Error: The file '{input_sto_file}' was not found.")
