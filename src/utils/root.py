import os
import logging


log = logging.getLogger(__name__)


def get_root_abs_path() -> str:
    '''
    Get absolute path of root directory of the project.
    Note this file must be in /project_root/src/utils/root.py
    '''
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../..'))

def get_temp_data_abs_path() -> str:
    return os.path.join(get_root_abs_path(), 'temp_data')

def create_temp_data_dir(path: str) -> str:
    '''
    Recursive function to creates dir inside of `temp_data` dir.
    Returns path to directory.
    '''
    dir = os.path.join(get_root_abs_path(), 'temp_data', path)
    log.info(f'Creating dir: {dir}')
    os.makedirs(dir, exist_ok=True)
    return dir

def create_temp_data_file(path: str, file_name: str) -> str:
    '''
    Recursive function to create file path and file inside of `temp_data` dir.
    Returns path to file.
    '''
    dir = create_temp_data_dir(path)
    file_path = os.path.join(dir, file_name)
    log.info(f'Creating file: {file_path}')
    with open(file_path, 'w') as f:
        pass
    return file_path


if __name__ == '__main__':
    pass
    # log.info(get_temp_data_abs_path())