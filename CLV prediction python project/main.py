from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from openpyxl import workbook
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Additional info; 'outlier_thresholds' function, is so useful function you can use it in your data preparation process, it finds out outlier_thresholds and equals it either up_limit or low_limit. They are simultaneously used with replace_with_thresholds function. With the help of these two function, We will equalize our outlier thresholds to determined low_limit and up_limit values without taking a long time.
#Where you can find the dataset https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
df_= pd.read_excel ('online_retail_II.xlsx', sheet_name= 'Year 2010-2011')
df=df_.copy()
#print(df)
#print(df.dtypes)
#df.shape
#print(df.describe())
#print(df.isnull().sum())

df.dropna(inplace=True )
#print(df)
#print(df.describe())
df.drop([ "StockCode", "Description", "Country"], axis=1, inplace=True)
print(df)
print(df.describe())
df = df[df['Quantity'] > 0]
print(df.describe())
print(df)

df['TotalPrice'] = df['Quantity'] * df['Price']
print(df.head())

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
 lambda date: (df['InvoiceDate'].max() - date.min()).days],
 'Invoice': lambda num: num.nunique(),
 'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
print(cltv_df.head(8))

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
print(cltv_df.head(5))

#Expressing "monetary value" as average earnings per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# selection of monetary values greater than zero
cltv_df = cltv_df[cltv_df["monetary"] > 0]
# Expression of "recency "and "T" in weekly terms
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
#frequency must be greater than 1.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
#After all trasformation ,lets check how our data looks like now.
print(cltv_df.head())

bgf = BetaGeoFitter(penalizer_coef=0.001)
#setting up the model
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
#fitting of all dataset

#Who are the 10 customers we expect the most to purchase in a week?
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
#1 = 1 week

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
#conditional_expected_number_of_purchases_up_to_time and predict are basically same.
print(cltv_df.head(5))

#Who are the 10 customers we expect the most to purchase in 1 month?
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df['expected_purc_1_month'] = bgf.predict(4,
 cltv_df['frequency'],
 cltv_df['recency'],
 cltv_df['T'])
#print(cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10))

#Sorted the variables according to 1month expected purchase prediction.

#What is the Expected Number of Sales of the Whole Company in 1 Month?
bgf.predict(4,cltv_df['frequency'],cltv_df['recency'],cltv_df['T']).sum()
#OUTPUT = 1777.1450731636987
#It is a transaction amount that is expected in the next month.

plot_period_transactions(bgf)
#plt.show()


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
#As I mentioned above, we only use two variables . We used 3 variables on BG-NBD model.
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                       cltv_df['monetary']).head(10)
ggf.conditional_expected_average_profit(cltv_df['frequency'],cltv_df['monetary'])
cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
#print(cltv_df['expected_average_profit'] .head(10))
#Let's sorted values from largest to smallest
#print(cltv_df.sort_values('expected_average_profit',ascending=False).head(10))

#Calculation of CLTV with BG-NBD and Gamma-Gamma Model
#Now we will mix both models and attain pure, analyzed CLTV values.

cltv = ggf.customer_lifetime_value(bgf,
 cltv_df['frequency'],
 cltv_df['recency'],
 cltv_df['T'],
 cltv_df['monetary'],
 time=3, # 3 Months
 freq='W', # Frequency of T ,in this case it is 'weekly'
 discount_rate=0.01)
#print(cltv.head(5))

#print(cltv.shape)
#OUTPUT = (2845,)
cltv = cltv.reset_index()
#print(cltv.head())

#print(cltv.sort_values(by="clv", ascending=False).head(10))

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
#merging our real dataset and cltv_df data.
#print(cltv_final)

#print(cltv_final.sort_values(by='clv', ascending=False).head(10))
#Sorting values by 'clv' variable.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#score between 0-1 ,1 is best 0 is worst. You can change by your wish, if you want you can score between 0-100 .
scaler.fit(cltv_final[['clv']])
cltv_final['scaled_clv'] = scaler.transform(cltv_final[['clv']])
#print(cltv_final.sort_values(by='scaled_clv', ascending=False).head())
#So let's observe it.

#Let's divide the customers into 4 groups:
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
#print(cltv_final.head())

print(cltv_final.sort_values(by="scaled_clv", ascending=False).head(10))