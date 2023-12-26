import sys
from contextlib import contextmanager


@contextmanager
def redirect_stdout_to_file(file_path, mode='w'):
    """
    使用上下文管理器重定向输出到指定文件。
    参数：
    - file_path: 输出文件的路径。
    - mode: 写入模式，'a' 为追加，'w' 为覆盖。
    """
    original_stdout = sys.stdout
    with open(file_path, mode) as file:
        sys.stdout = file
        yield
        sys.stdout = original_stdout
