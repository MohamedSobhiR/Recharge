import joblib
import numpy as np 

model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')

#Index(['Year', 'Month', 'Week', 'Day', 'Week_Day', 'RechargeCount'])
year = int(input(("Your Predicted Year : ")))

month = int(input(("Your Predicted Month : ")))

week = int(input(("Your Predicted Week : ")))

day = int(input(("Your Predicted Day : ")))

week_day = int(input(("Your Predicted Week_Day : ")))

rechargeCount = int(input(("Your Predicted RechargeCount : ")))

pre_input = {'Year':year, 'Month':month, 'Week':week, 'Day':day, 'Week_Day':week_day, 'RechargeCount':rechargeCount}

custom_data = np.array(list(pre_input.values()))

custom_data = scaler.transform([custom_data])

prediction = model.predict(custom_data)

print(f'Total Recharge amount is :  {prediction[0]}')