import tarfile

if __name__ == "__main__":
    with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar: tar.extractall()