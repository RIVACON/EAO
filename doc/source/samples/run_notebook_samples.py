# Check if all the notebooks collected in the documentation run successfully
import sys
import os
import unittest
import warnings
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
warnings.filterwarnings("ignore")


def execute_notebook(notebook_file: Path, 
                        notebook_folder: Path,
                            timeout: int = 600) -> None:

    """ Simple function to RUN a Jupyter notebook.
    If all the cells are executed, returns True, otherwise False.
    The notebook is specified in the argument notebook_path,
    and is supposed to last no more than timeout seconds.
    """

    try:
        # Load the notebook
        notebook_path = os.path.join(notebook_folder, notebook_file)
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Configure and execute the notebook
        ep = ExecutePreprocessor(timeout=timeout)
        # Run notebooks as if they were in their original folder
#       ep.preprocess(nb,{'metadata': {'path': './docs/source/notebooks'}})
        ep.preprocess(nb,{'metadata': {'path': notebook_folder}})

    except FileNotFoundError:
        msg = f"Notebook file not found: {notebook_path}"
        print(msg)
        return False

    except CellExecutionError as e:
        print(e)
        msg = f"Error executing notebook {notebook_path}"
        msg += f"\nPlease load it manually for debugging."
        print(msg)
        return False

    print("OK")
    return True
    #---

def test_notebook_execution():
    """ Run all the notebooks and return True if there is no error at all.
    """
    # Use the previous function to run every notebook of interest
    notebook_folder = os.path.join("empty_notebook")
    my_notebooks = ["import_eao.ipynb"]
    for name in my_notebooks:
        print(f"Testing {my_notebooks}")
        res = execute_notebook(name, notebook_folder)
        if res is False:
            return False
    return True

if __name__ == "__main__":
    res = test_notebook_execution()
    print(res)
    if res:
        print("Running Samples: SUCCESS")
        sys.exit(0)
    else:
        print("Running Samples: FAIL")
        sys.exit(1)