import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


# Carregamento dos dados

df = read_csv(
    'shampoo-sales.csv'
)
df['Month'] = '190' + df['Month']
df['Month'] = pd.to_datetime(df['Month'])

df = df.set_index('Month')
print(df.head())
print(df.info())

series = df['Sales']

# ARIMA(5, 2, 1)

model = ARIMA(series, order=(5, 2, 1))
model_fit = model.fit()

print(model_fit.summary())

# Análise dos resíduos

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

#residuals.plot(kind='kde')
#pyplot.show()

print(residuals.describe())

# ---------------------------------------------------------------#
# Autocorrelação Manual


sales = df["Sales"].tolist()
media = sum(sales) / len(sales)
# print(sales)
# print(media)

autocorrelations = {}
for lag in range(36):
    x1 = sales[:len(sales)-lag]
    x2 = sales[lag:]
    num_r_k = 0
    den_r_k = 0
    for n in range(len(x1)):
        num_r_k += (x1[n]-media) * (x2[n]-media)
    for n in range(len(sales)):
        den_r_k += (sales[n] - media) ** 2
    r_k = num_r_k/den_r_k
    autocorrelations[lag] = r_k

lags = autocorrelations.keys()
rks = autocorrelations.values()

# print(autocorrelations)

# -------------------------------------------------------------------
# Walk-Forward

X = series.values

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:]

history = [x for x in train]
predictions = []

for t in range(len(test)):

    model = ARIMA(history, order=(5, 2, 1))
    model_fit = model.fit()

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]
    history.append(obs)

    print(f'predicted={yhat:.3f}, expected={obs:.3f}')

rmse = mean_squared_error(test, predictions) ** 0.5

print(f'Test RMSE: {rmse:.3f}')

pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()

result = adfuller(series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

print("Série original:")
print(adfuller(series)[1])

print("\nPrimeira diferença:")
print(adfuller(series.diff().dropna())[1])

print("\nSegunda diferença:")
print(adfuller(series.diff().diff().dropna())[1])

# pyplot.plot(df["Month"], df["Sales"])
# pyplot.show()
# pyplot.plot(lags, rks)
# pyplot.show()

"""                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Sales   No. Observations:                   36
Model:                 ARIMA(5, 1, 0)   Log Likelihood                -198.485
Date:                Wed, 02 Sep 2026   AIC                            408.969
Time:                        15:47:05   BIC                            418.301
Sample:                    01-01-1901   HQIC                           412.191
                         - 12-01-1903                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.9014      0.247     -3.647      0.000      -1.386      -0.417
ar.L2         -0.2284      0.268     -0.851      0.395      -0.754       0.298
ar.L3          0.0747      0.291      0.256      0.798      -0.497       0.646
ar.L4          0.2519      0.340      0.742      0.458      -0.414       0.918
ar.L5          0.3344      0.210      1.593      0.111      -0.077       0.746
sigma2      4728.9608   1316.021      3.593      0.000    2149.607    7308.314
===================================================================================
Ljung-Box (L1) (Q):                   0.61   Jarque-Bera (JB):                 0.96
Prob(Q):                              0.44   Prob(JB):                         0.62
Heteroskedasticity (H):               1.07   Skew:                             0.28
Prob(H) (two-sided):                  0.90   Kurtosis:                         2.41
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
                0
count   36.000000
mean    21.936145
std     80.774430
min   -122.292030
25%    -35.040859
50%     13.147219
75%     68.848286
max    266.000000

predicted=343.272, expected=342.300
predicted=293.330, expected=339.700
predicted=368.669, expected=440.400
predicted=335.045, expected=315.900
predicted=363.220, expected=439.300
predicted=357.645, expected=401.300
predicted=443.048, expected=437.400
predicted=378.366, expected=575.500
predicted=459.415, expected=407.600
predicted=526.891, expected=682.000
predicted=457.231, expected=475.300
predicted=672.915, expected=581.300
predicted=531.541, expected=646.900
Test RMSE: 89.021
"""
# ARIMA(5, 1, 1): Test  RMSE: 76.952