import pandas as pd
import joblib
import os

mails=[]
for file_name in os.listdir('test'):
    with open('test/'+file_name) as file:
        mails.append(file.read())

# labels=np.array(([1]*25 + [0]*25)*4)

count_vec=joblib.load('count_vec.pkl')
log_reg=joblib.load('log_reg.pkl')

mail_dataset=count_vec.transform(mails)
predictions=log_reg.predict(mail_dataset).flatten()

# print((predictions==labels).mean()*100)
# print(predictions)

pd.Series(predictions.astype(int)).to_csv('predictions.csv', index=False, header=False)




