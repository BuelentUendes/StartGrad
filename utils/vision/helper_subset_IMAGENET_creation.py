import os
import random
import shutil


def count_folders_and_files(directory):
    folder_counts = {}

    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        folder_counts[folder_name] = {'folders': len(dirs), 'files': len(files)}

    return folder_counts


def create_subset_of_images(src_directory, dest_directory, subset_size=5, seed=123):
    if os.path.exists(dest_directory):
        print(f"The destination directory '{dest_directory}' already exists. No new subset will be created.")
        return

    # Create the destination directory if it does not exist
    os.makedirs(dest_directory, exist_ok=True)

    # Iterate through each folder in the source directory
    for folder in os.listdir(src_directory):
        src_folder_path = os.path.join(src_directory, folder)
        dest_folder_path = os.path.join(dest_directory, folder)

        # Ensure the current item is a directory
        if os.path.isdir(src_folder_path):
            # Create the corresponding directory in the destination
            os.makedirs(dest_folder_path, exist_ok=True)

            # List all files in the source folder
            all_files = os.listdir(src_folder_path)

            # Select a random subset of files
            selected_files = random.sample(all_files, min(subset_size, len(all_files)))

            # Copy the selected files to the destination folder
            for file in selected_files:
                src_file_path = os.path.join(src_folder_path, file)
                dest_file_path = os.path.join(dest_folder_path, file)
                shutil.copy(src_file_path, dest_file_path)


