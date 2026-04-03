import pandas as pd
from sklearn.model_selection import train_test_split
import os

SEED = 42
TRAIN_RATIO = 0.10
VAL_RATIO = 0.07
TEST_RATIO = 0.83

data_dir = os.path.join(os.path.dirname(__file__), "..", ".dataset")
input_path = os.path.join(data_dir, "officeqa_easy.csv")
df = pd.read_csv(input_path)

print(f"Total samples: {len(df)}")

train_val_ratio = TRAIN_RATIO + VAL_RATIO
train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=SEED)
val_relative = VAL_RATIO / train_val_ratio
train_df, val_df = train_test_split(train_df, test_size=val_relative, random_state=SEED)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

train_df.to_csv(os.path.join(data_dir, "officeqa_easy_train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "officeqa_easy_val.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "officeqa_easy_test.csv"), index=False)

print("Saved to .dataset/officeqa_easy_{train,val,test}.csv")
