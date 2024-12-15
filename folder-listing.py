import os

def list_files_hierarchically(start_path='.', max_depth=3, max_dirs=10, max_files=10):
    """
    Print the directory structure starting from the given path in a hierarchical format,
    limiting the depth and the number of subdirectories and files displayed.

    Parameters:
    - start_path (str): The root directory to start listing from.
    - max_depth (int): The maximum depth to traverse.
    - max_dirs (int): The maximum number of subdirectories to display per directory.
    - max_files (int): The maximum number of files to display per directory.
    """
    def traverse(current_path, current_depth):
        if current_depth > max_depth:
            return
        
        indent = '│   ' * current_depth
        folder_name = os.path.basename(current_path) or start_path
        prefix = "📁 " if current_depth == 0 else "└── 📁 "
        print(f"{indent}{prefix}{folder_name}")
        
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            print(f"{indent}    [Permission Denied]")
            return
        
        dirs = [entry for entry in entries if os.path.isdir(os.path.join(current_path, entry))]
        files = [entry for entry in entries if os.path.isfile(os.path.join(current_path, entry))]
        
        # Limit the number of subdirectories displayed
        for idx, directory in enumerate(dirs[:max_dirs]):
            traverse(os.path.join(current_path, directory), current_depth + 1)
            if idx == max_dirs - 1 and len(dirs) > max_dirs:
                print(f"{indent}    └── ... and {len(dirs) - max_dirs} more folders")
        
        # Prepare indentation for files
        file_indent = '│   ' * (current_depth + 1)
        
        # Limit the number of files displayed
        for file in sorted(files)[:max_files]:
            print(f"{file_indent}└── 📄 {file}")
        if len(files) > max_files:
            print(f"{file_indent}└── ... and {len(files) - max_files} more files")
    
    traverse(os.path.abspath(start_path), 0)

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    list_files_hierarchically(current_path, max_depth=3, max_dirs=30, max_files=30)
