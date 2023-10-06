import tarfile
import glob
import subprocess

if __name__ == "__main__":

    doi = '10.5281/zenodo.8406322 '
    output_dir = './EMVD/'

    #############
    # GET ZIP FILES FROM ZENODO

    command = ['zenodo_get', doi, '-o', output_dir]
    subprocess.run(command, check=True)

    ############
    # EXTRACT ZIP FILES

    # Get a list of all .tar.gz files in the directory
    tar_files = glob.glob(output_dir + '/*.tar.xz')

    # Extract each .tar.gz file
    for tar_file in tar_files:
        with tarfile.open(tar_file, 'r:xz') as tar:
            tar.extractall(output_dir)

