from pathlib import Path
import subprocess
from typing import Dict
from abc import ABC, abstractmethod


class BaseScriptRunner(ABC):
    """
    A base class for script runners.
    """

    def __init__(self, script_path: Path, args: Dict[str, str], output_file_path: Path = None):
        """
        Initialize the BaseScriptRunner with a script path and arguments.

        :param script_path: Path to the script to be run.
        :param args: A dictionary of argument names and values.
        :param output_file_path: Path to the log file.
        """
        self.script_path = script_path
        self.args = args
        self.log_file_path = output_file_path

    @abstractmethod
    def run_script(self):
        """
        Abstract method to execute the script. This method should be implemented in derived classes.
        """
        pass


class PythonScriptRunner(BaseScriptRunner):
    """
    A class to run a python script with given arguments using subprocess.
    """

    def run_script(self):
        """
        Execute the python script with the specified arguments using "subprocess.run".

        由于 "subprocess.run" 默认是阻塞性的，它会等待启动的进程结束后再继续执行。
        """
        # Construct the command with arguments
        cmd = ["python", str(self.script_path)] + [f"--{k}={v}" for k, v in self.args.items()]
        print(f"PythonScriptRunner.run_script -> Running command: {cmd}")

        # Open the output file
        if self.log_file_path:
            with open(self.log_file_path, 'w') as output_file:
                # Execute the command using subprocess and redirect output to the file
                subprocess.run(cmd, stdout=output_file, stderr=output_file)
        else:
            # Execute the command using subprocess
            subprocess.run(cmd)