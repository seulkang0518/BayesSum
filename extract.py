import subprocess
import os

# --- Define your files ---
# The unaligned FASTA file you want to align
input_fasta = "PF12740.fasta" 
# The name for the output file with the final alignment
output_alignment = "PF12740_aligned.fasta"

print(f"Running MUSCLE on {input_fasta}...")
print("This may take a few minutes...")

# This is the modern command structure for MUSCLE v5+
command = [
    "muscle",
    "-align", input_fasta,
    "-output", output_alignment
]

try:
    # Run the command
    subprocess.run(command, check=True, capture_output=True, text=True)
    print("Alignment complete!")
    print(f"Aligned sequences saved to {output_alignment}")

except FileNotFoundError:
    print("Error: The 'muscle' command was not found.")
    print("Please make sure MUSCLE is installed and accessible in your system's PATH.")

except subprocess.CalledProcessError as e:
    print(f"An error occurred while running MUSCLE (return code {e.returncode}):")
    print(f"Error message: {e.stderr}")