import numpy as np

#FEATURE ENGINEERING

#Extracting hour from the scheduled time
OTP_Copy['hour'] = OTP_Copy['scheduled_time'].dt.hour
OTP_Copy[["scheduled_time","hour"]].tail()

#Similarly extracting Day of the week, month, weekend, Rush Hour, Season, OTP (1,0)
OTP_Copy['day_of_week'] = OTP_Copy['date'].dt.day_name()

OTP_Copy['month'] = OTP_Copy['date'].dt.month

OTP_Copy["weekend"] = np.where(OTP_Copy["date"].dt.dayofweek >=5,1,0)

OTP_Copy["rush_hour"] = np.where(
    ((OTP_Copy["hour"]>=7)&(OTP_Copy["hour"]<=10))|
    ((OTP_Copy["hour"]>=16)&(OTP_Copy["hour"]<=19)),
    1,
    0
)

def season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Fall"

OTP_Copy["season"] = OTP_Copy["month"].apply(season)

OTP_Copy["otp_binary"] = np.where(
    OTP_Copy["delay_minutes"]<=5,
    1,
    0
)

OTP_Copy[["date","day_of_week","month","weekend","rush_hour","season","otp_binary"]].head()
