import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot


df = read_csv(
    'shampoo-sales.csv',
)
df['Month'] ='190'+df['Month']
df['Month'] = pd.to_datetime(df['Month'])
# print(df.head())
# print(df.info())

sales = df["Sales"].tolist()
media = sum(sales) / len(sales)
# print(sales)
# print(media)

autocorrelations = {}
for lag in range (36):
    x1 = sales[:len(sales)-lag]
    x2 = sales[lag:]
    num_r_k=0
    den_r_k=0
    for n in range(len(x1)):
        num_r_k += (x1[n]-media) * (x2[n]-media)
    for n in range(len(sales)):
        den_r_k += (sales[n] - media) ** 2
    r_k = num_r_k/den_r_k
    autocorrelations[lag] = r_k

lags = autocorrelations.keys()
rks = autocorrelations.values()

# pyplot.plot(df["Month"], df["Sales"])
# pyplot.show()
pyplot.plot(lags, rks)
pyplot.show()


