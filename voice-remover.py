import os
import shutil
import tempfile
from audio_separator.separator import Separator

def process_audio(input_file, models, final_output_name="output_final.wav"):
    """
    Process an audio file using a sequence of machine learning models, keeping the input intact
    and storing intermediate files in a temporary folder. Allows specifying which output file to use.

    Args:
        input_file (str): Path to the input audio file.
        models (list of tuple): List of (model filename, output index) to apply in sequence.
        final_output_name (str): Name of the final output file.

    Returns:
        str: Path to the final processed file.
    """
    # Initialize the Separator
    separator = Separator()
    
    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    current_file = input_file

    try:
        # Apply models sequentially
        for i, (model, output_index) in enumerate(models):
            separator.load_model(model)
            output_files = separator.separate(current_file)
            print(f"Output files from {model}: {output_files}")
            
            # Select the specified output file
            if output_index >= len(output_files):
                raise ValueError(f"Invalid output index {output_index} for model {model}.")
            current_file = output_files[output_index]

            # Move intermediate files to the temp directory
            for file in output_files:
                temp_path = os.path.join(temp_dir, os.path.basename(file))
                shutil.move(file, temp_path)

            current_file = os.path.join(temp_dir, os.path.basename(current_file))
            print(f"Processed with {model}: {current_file}")

        # Save the final output with a clean name in the same directory as the input file
        final_output_path = os.path.join(os.path.dirname(input_file), final_output_name)
        shutil.copy(current_file, final_output_path)

        print(f"Final output saved as: {final_output_path}")
        return final_output_path

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} deleted.")

# Input file and models to process
input_audio = 'dirty.wav'
models_to_apply = [
    ("Kim_Vocal_2.onnx", 0),
]

# Process the audio file with a user-friendly final output name
final_output = process_audio(input_audio, models_to_apply, final_output_name="no_voice.wav")

print(f"Processing complete! Final output file: {final_output}")
