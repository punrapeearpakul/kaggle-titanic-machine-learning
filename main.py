import pandas as pd

TRAIN_DATA_PATH = r"data\train.csv"
TEST_DATA_PATH = r"data\test.csv"
INDEX_COL = "PassengerId"

# ====== Data Loader ====
train_data = pd.read_csv(TRAIN_DATA_PATH,index_col=INDEX_COL )
test_data = pd.read_csv(TEST_DATA_PATH,index_col=INDEX_COL)