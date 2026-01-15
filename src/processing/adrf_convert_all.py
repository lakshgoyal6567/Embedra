import argparse
import os
import glob
import subprocess
import sys

# Helper function to run commands (similar to project_name.py's run_command)
def run_sub_command(command_parts, cwd=None): # command_parts is now a list of strings
    process = subprocess.run(command_parts, cwd=cwd, capture_output=True, text=True) # Removed shell=True
    if process.stdout:
        print(process.stdout)
    if process.stderr:
        sys.stderr.write(process.stderr)
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="ADRF Multi-modal Converter: Process all raw data types.")
    parser.add_argument("--input_dir", type=str, default="data/raw_data", help="Path to directory containing all raw data")
    
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_dir)

    if not os.path.exists(input_path):
        print(f"Error: Input directory {input_path} does not exist. Please create it and add your data.")
        sys.exit(1)

    print(f"Scanning input directory: {input_path}")

    # Categorize files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif', '.tiff', '.tif')
    doc_extensions = ('.txt', '.pdf', '.docx')

    found_image_files = False
    found_doc_files = False

    for root, _, files in os.walk(input_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in image_extensions:
                found_image_files = True
            elif file_extension in doc_extensions:
                found_doc_files = True
            # Add audio/video checks here in the future
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Process Images
    if found_image_files:
        print("\n--- Starting Image Processing ---")
        image_extensions_str = ",".join(image_extensions)
        cmd_parts = [
            "python", 
            "adrf_convert.py", 
            "--input_dir", input_path, 
            "--file_types", image_extensions_str
        ]
        return_code = run_sub_command(cmd_parts, cwd=script_dir)
        if return_code != 0:
            print("Image processing failed.")
        else:
            print("Image processing complete.")
        print("--- Image Processing Finished ---\n")
    else:
        print("No image files found. Skipping image processing.")

    # Process Documents
    if found_doc_files:
        print("\n--- Starting Document Processing ---")
        doc_extensions_str = ",".join(doc_extensions)
        cmd_parts = [
            "python", 
            "adrf_convert_docs.py", 
            "--input_dir", input_path, 
            "--file_types", doc_extensions_str
        ]
        return_code = run_sub_command(cmd_parts, cwd=script_dir)
        if return_code != 0:
            print("Document processing failed.")
        else:
            print("Document processing complete.")
        print("--- Document Processing Finished ---\n")
        
    print("\n--- All requested processing types completed ---")

if __name__ == "__main__":
    main()
