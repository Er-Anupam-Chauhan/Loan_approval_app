from config import * 

def load_data(path):
    """
    Load data from a CSV file and log the operation.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    
    df = pd.read_csv(path)
    logger.info("Data loading finished !")
    return df