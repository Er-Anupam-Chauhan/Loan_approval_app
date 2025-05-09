import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import logger
from logger import get_logger

logger = get_logger("logs/training.log")


path_to_the_data = "D:/Projects/Loan_approval_classification/data/train_u6lujuX_CVtuZ9i.csv"
model = LogisticRegression()
