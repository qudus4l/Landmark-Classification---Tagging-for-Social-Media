import tarfile
import glob
import datetime
import subprocess
import os

def create_submit_pkg():
    # Get a list of all files and directories in the current directory
    all_files = glob.glob("*")

    # Generate HTML files from the notebooks
    notebooks = glob.glob("*.ipynb")
    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"
        print(f"executing: {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    # Get a list of all HTML files
    html_files = glob.glob("*.htm*")

    # Generate a timestamped filename for the submission package
    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h") + "m"
    outfile = f"submission_{now}.tar.gz"
    print(f"Adding files to {outfile}")

    # Create the tarball and add all files and directories to it
    with tarfile.open(outfile, "w:gz") as tar:
        for name in (all_files + html_files):
            print(name)
            tar.add(name)

    print("")
    msg = f"Done. Please submit the file {outfile}"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))

if __name__ == "__main__":
    create_submit_pkg()
