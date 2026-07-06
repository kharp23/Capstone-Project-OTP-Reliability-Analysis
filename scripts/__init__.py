import pandas as pd

files = [
    r"file1path.csv",
    r"file2path.csv",
    r"file3path.csv",
    r"file4path.csv",
    r"file5path.csv"
]

dfs = [pd.read_csv(file) for file in files]

OTP_df = pd.concat(dfs, ignore_index=True)

print("Merged Dataset Shape:", OTP_df.shape)

OTP_df.head()

OTP_df.to_csv("OTP_df.csv", index=False)

print("Merged file saved as merged_dataset.csv")
