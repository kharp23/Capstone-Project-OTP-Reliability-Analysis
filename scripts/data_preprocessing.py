import pandas as pd
#Duplicate rows check
duplicates= OTP_df.duplicated().sum()
duplicates
#Removing the duplicate rows
OTP_df.drop_duplicates(inplace=True)


#MISSING VALUE TREATMENT
OTP_df['delay_minutes'] = OTP_df['delay_minutes'].fillna(
    OTP_df['delay_minutes'].median()
)

OTP_df['status'] = OTP_df['status'].fillna('Unknown')
OTP_df['line'] = OTP_df['line'].fillna('Unknown')

#TIMESTAMP-PARSING- Convert time columns
OTP_df['scheduled_time'] = pd.to_datetime(OTP_df['scheduled_time'])
OTP_df['actual_time'] = pd.to_datetime(OTP_df['actual_time'])
OTP_df['date'] = pd.to_datetime(OTP_df['date'])

OTP_df.dtypes
