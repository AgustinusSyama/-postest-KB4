# -postest-KB4
NAMA AGUSTINUS SYAMA
NIM 2009106150


Posttest 4 Preprocessing
Ahmad Zidan Maulidinnur
2009106018
Kecerdasan Buatan A2 2020
import pandas as pd
import numpy as np
df = pd.read_csv('Stroke.csv')
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
df.head()
id	gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke
0	9046	Male	67.0	0	1	Yes	Private	Urban	228.69	36.6	formerly smoked	1
1	51676	Female	61.0	0	0	Yes	Self-employed	Rural	202.21	NaN	never smoked	1
2	31112	Male	80.0	0	1	Yes	Private	Rural	105.92	32.5	never smoked	1
3	60182	Female	49.0	0	0	Yes	Private	Urban	171.23	34.4	smokes	1
4	1665	Female	79.0	1	0	Yes	Self-employed	Rural	174.12	24.0	never smoked	1
Membagi dataset menjadi training set dan testing set dengan proporsi 70:30
x = df.iloc[:,:-1] # Target
y = df.iloc[:,-1] # Feature
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print("Dimensi X_train : ", x_train.shape)
print("Dimensi X_train : ", x_test.shape)
print("Dimensi y_train : ", y_train.shape)
print("Dimensi y_test  : ", y_test.shape)
Dimensi X_train :  (3577, 11)
Dimensi X_train :  (1533, 11)
Dimensi y_train :  (3577,)
Dimensi y_test  :  (1533,)

print("Nilai Data setelah scaling : ")
print(Scaled_data)
print("\n Nilai standar deviasi: ", np.std(Scaled_data))
Nilai Data setelah scaling : 
[[0.81689453 1.         0.80126489 1.        ]
 [0.74365234 0.         0.67902317 1.        ]
 [0.97558594 1.         0.23451205 1.        ]
 ...
 [0.42626953 0.         0.12865848 0.        ]
 [0.62158203 0.         0.51320284 0.        ]
 [0.53613281 0.         0.13922999 0.        ]]

 Nilai standar deviasi:  0.3035009747787989
Melakukan standarisasi pada dataset
copy_dataset = df[[ 'age', 'heart_disease', 'avg_glucose_level', 'stroke' ]]

print("Nilai Data Sebelum scaling : ")
print(copy_dataset)
Nilai Data Sebelum scaling : 
       age  heart_disease  avg_glucose_level  stroke
0     67.0              1             228.69       1
1     61.0              0             202.21       1
2     80.0              1             105.92       1
3     49.0              0             171.23       1
4     79.0              0             174.12       1
...    ...            ...                ...     ...
5105  80.0              0              83.75       0
5106  81.0              0             125.20       0
5107  35.0              0              82.99       0
5108  51.0              0             166.29       0
5109  44.0              0              85.28       0

[5110 rows x 4 columns]
from sklearn.preprocessing import StandardScaler

StandardScaler = StandardScaler()
Scaled_data = StandardScaler.fit_transform(copy_dataset)

print("Nilai Data setelah scaling : ")
print(Scaled_data)
print("\n Nilai standar deviasi: ", np.std(Scaled_data))
Nilai Data setelah scaling : 
[[ 1.05143428  4.18503199  2.70637544  4.41838074]
 [ 0.78607007 -0.2389468   2.12155854  4.41838074]
 [ 1.62639008  4.18503199 -0.0050283   4.41838074]
 ...
 [-0.36384151 -0.2389468  -0.51144264 -0.22632726]
 [ 0.34379639 -0.2389468   1.32825706 -0.22632726]
 [ 0.03420481 -0.2389468  -0.46086746 -0.22632726]]

 Nilai standar deviasi:  1.0
Melakukan Data cleaning pada data dengan nilai null. Mengganti nilai null sesuai ketentuan. (bilangan bulat : median/modus, bilangan desimal : mean, tulisan : modus)
print("Jumlah record yang memiliki nilai null: ")
print(df.isna().sum())

tipeData_creditProduct = df.dtypes['Residence_type']
print("\nTipe data atribut yg memiliki nilai null: ")
print(tipeData_creditProduct)
Jumlah record yang memiliki nilai null: 
id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
dtype: int64

Tipe data atribut yg memiliki nilai null: 
object
from sklearn.impute import SimpleImputer
SimpleImputer = SimpleImputer(strategy='most_frequent' )
df["Credit_Product"] = SimpleImputer.fit_transform(df[["Residence_type"]])

print("\nJumlah nilai null setelah menggunakan SimpleImputer: ")
print(df.isna().sum())
Jumlah nilai null setelah menggunakan SimpleImputer: 
id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
Credit_Product         0
dtype: int64
Melakukan Data cleaning pada data dengan nilai duplikat. (Jika tidak ada nilai duplikat pada dataset, maka buatlah menjadi ada)
# menampilkan nilai duplikat dataset
print("Jumlah nilai duplikat pada dataset :", df.duplicated().sum())
df[df.duplicated()]
Jumlah nilai duplikat pada dataset : 0
id	gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke	Credit_Product
# membuat nilai duplikat menjadi ada
df = df.append(df.iloc[0:100])
print("Jumlah setelah membuat nilai duplikat     :", df.duplicated().sum())

# melakukan data cleaning dengan nilai duplikat
df.drop_duplicates(inplace=True)
print("Nilai duplikat setelah dilakukan cleaning :", df.duplicated().sum())
Jumlah setelah membuat nilai duplikat     : 100
Nilai duplikat setelah dilakukan cleaning : 0
C:\Users\Zidan\AppData\Local\Temp\ipykernel_2976\1971219378.py:2: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
  df = df.append(df.iloc[0:100])
Mengganti tipe data salah satu attribute angka
print("Tipe data sebelum diubah :", df.dtypes['heart_disease'])
df['heart_disease'] = df['heart_disease'].astype('float64')
print("Tipe data setelah diubah :", df.dtypes['heart_disease'])
Tipe data sebelum diubah : int64
Tipe data setelah diubah : float64
Melakukan one hot encoding pada dataset yang dimiliki
from sklearn.preprocessing import OneHotEncoder
print("Sebelum dilakukan Encoding")
df.head()
Sebelum dilakukan Encoding
id	gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke	Credit_Product
0	9046	Male	67.0	0	1.0	Yes	Private	Urban	228.69	36.6	formerly smoked	1	Urban
1	51676	Female	61.0	0	0.0	Yes	Self-employed	Rural	202.21	NaN	never smoked	1	Rural
2	31112	Male	80.0	0	1.0	Yes	Private	Rural	105.92	32.5	never smoked	1	Rural
3	60182	Female	49.0	0	0.0	Yes	Private	Urban	171.23	34.4	smokes	1	Urban
4	1665	Female	79.0	1	0.0	Yes	Self-employed	Rural	174.12	24.0	never smoked	1	Rural
encoder = OneHotEncoder(sparse=False)
print("Melakukan OneHotEncoding pada atribut Residence_type")
Residence_type_Encoder = encoder.fit_transform(df[["Residence_type"]])
Residence_type = pd.DataFrame(Residence_type_Encoder)

df = df.join(Residence_type)
df.head()
Melakukan OneHotEncoding pada atribut Residence_type
id	gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke	Credit_Product	0	1
0	9046	Male	67.0	0	1.0	Yes	Private	Urban	228.69	36.6	formerly smoked	1	Urban	0.0	1.0
1	51676	Female	61.0	0	0.0	Yes	Self-employed	Rural	202.21	NaN	never smoked	1	Rural	1.0	0.0
2	31112	Male	80.0	0	1.0	Yes	Private	Rural	105.92	32.5	never smoked	1	Rural	1.0	0.0
3	60182	Female	49.0	0	0.0	Yes	Private	Urban	171.23	34.4	smokes	1	Urban	0.0	1.0
4	1665	Female	79.0	1	0.0	Yes	Self-employed	Rural	174.12	24.0	never smoked	1	Rural	1.0	0.0
print("Mengubah nama kolom")
df = df.rename(columns={
    0 : 'Residence_type_No',
    1 : 'Residence_type_Yes'
})
df.head(10)
Mengubah nama kolom
id	gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke	Credit_Product	Residence_type_No	Residence_type_Yes
