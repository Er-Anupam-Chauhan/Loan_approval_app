from config import * 
from load_data import * 
from preprocessing import * 
from model_training import *

logger.info(" --- Pipeline Start ---")

df = load_data(path_to_the_data)

df = preprocess_main(df)

model_training(model,df)

logger.info(" --- Pipeline End ---")
