import os


def get_project_root() -> str:
    """Returns the root directory of the project."""
    current_script_path = os.path.abspath(__file__)
    src_directory = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_directory)
    return project_root
