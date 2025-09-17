# Check if all the notebooks collected in the documentation run successfully
import sys
import os
import warnings
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
warnings.filterwarnings("ignore")

### do not evaluate black list of notebooks (e.g. with long run-time)
blacklist = ['large_problems.ipynb']


def execute_notebook(notebook_file: Path, 
                     notebook_folder: Path,
                     kernel_name = "eao_env",
                     timeout: int = 1800) -> bool:

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
        ep = ExecutePreprocessor(timeout=timeout,
                                 kernel_name = kernel_name)
        # Run notebooks as if they were in their original folder
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

    print(f"Notebook {notebook_path}: SUCCESS")
    return True
    #---


def find_all_notebooks():
    '''
    Visit the folder doc/source/samples and identifies all the
    existing notebooks together with their folders.
    Retuns the list of pairs (folder, notebook_file).
    This is important since all the notebooks should be run as
    if they where in their original folder, therefore the tracking.
    '''
    # Assume to work in the following path
    path_all_notebooks = os.path.join("doc", "source", "samples")
    all_folders = os.listdir(path_all_notebooks)
    # In each directory, seach for notebooks, store (folder,file)
    pairs = []
    for folder in all_folders:
        path_folder = os.path.join(path_all_notebooks, folder)
        # if it is a folder, then check notebooks
        if os.path.isdir(path_folder):
            print(f"Checking folder {path_folder}")
            # Check if there are files ending with ipynb
            for file in os.listdir(path_folder):
                if len(str(file)) > 6:
                    if str(file)[-6:] == ".ipynb":
                        # Append to the pair list
                        ### do not evaluate black list of notebooks (with long run-time)
                        if file not in blacklist:
                            print(f"\t{file}")
                            print(f"{path_folder}, {file}")
                            pairs.append((path_folder, file))
                        else:
                            print(f"ignored \t{file}")

        else:
            print(f"File {path_folder} not a folder: SKIP")
    return pairs


def test_notebook_execution():
    """ Run all the notebooks and return True if there is no error at all.
    """
    # Use the previous function to run every notebook of interest
    folder_file_pairs = find_all_notebooks()
    tot = len(folder_file_pairs)
    print(f"Total of {tot} notebooks to run.")
    fail = []
    # Use the kernel with the same name of conda environment
    my_kernel = os.getenv("CONDA_DEFAULT_ENV")
    if not my_kernel:
        print("Unable to load a kernel! Sure that you use conda?")
        return False

    for path_folder, filename in folder_file_pairs:
        full_file =os.path.join(path_folder, filename)
        print(f"Testing: {full_file}")
        #input("OK? (TO BE REMOVED)")
        res = execute_notebook(filename, path_folder, my_kernel)
        if res is False:
            print(f"Notebook: {full_file}: FAIL")
            fail.append(full_file)

    if len(fail) > 0:
        print("The following notebooks FAILED:")
        for fullfile in fail:
            print(fullfile)
        return False
    return True
#---

if __name__ == "__main__":
    folder_file_pairs = find_all_notebooks()
    res = test_notebook_execution()
    print(res)
    if res:
        print("Running Samples: SUCCESS")
        sys.exit(0)
    else:
        print("Running Samples: FAIL")
        sys.exit(1)