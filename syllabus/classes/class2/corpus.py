# Import Module
import os

# Corpus loader, returning a list of strings
def corpus_loader(folder_path: str) -> list:
    """
    A corpus loader function which takes in a path to a 
    folder and returns a list of strings.
    """
    # Creating empty list
    documents = []

    # Iterating through files, opening and appending the content to the list 
    for file in os.listdir(folder_path):
        # Specifying file path
        file_path = os.path.join(folder_path, file)

        # Open and read file
        with open(file_path, encoding="utf-8") as f:
            document = f.read()
            documents.append(document)
    
    return documents

########

# Potentially: create corpus class (corpus(self, folder_path))