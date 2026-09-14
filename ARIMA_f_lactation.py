import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error


# Carregamento dos dados

df = read_csv(
    'curva_lactacao_vaca_1.csv'
)
df = df.drop(columns="animal_id")
df = df.drop(columns="lact_num")
df = df.drop(columns="dim")
df["date"] = pd.to_datetime(df["date"])

df = df.set_index("date")
print(df.head())
print(df.info())

series = df["yield_kg"]

df["yield_kg"].plot()
pyplot.show()

# ---------------------------------------------------------------#
# Autocorrelação Manual


values = df["yield_kg"].tolist()
media = sum(values) / len(values)
print(values)
print(media)

autocorrelations = {}
for lag in range(len(values)):
    x1 = values[:len(values)-lag]
    x2 = values[lag:]
    num_r_k = 0
    den_r_k = 0
    for n in range(len(x1)):
        num_r_k += (x1[n]-media) * (x2[n]-media)
    for n in range(len(values)):
        den_r_k += (values[n] - media) ** 2
    r_k = num_r_k/den_r_k
    autocorrelations[lag] = r_k

lags = autocorrelations.keys()
rks = autocorrelations.values()

print(autocorrelations)

pyplot.plot(lags, rks)
pyplot.show()

# ARIMA(p, d, q) ---------------------------------------------------------------------------

model = ARIMA(series, order=(4, 1, 0))
model_fit = model.fit()

print(model_fit.summary())

# Análise dos resíduos

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

residuals.plot(kind='kde')
pyplot.show()

print(residuals.describe())


# -------------------------------------------------------------------
# Walk-Forward

X = series.values

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:]

history = [x for x in train]
predictions = []

for t in range(len(test)):

    model = ARIMA(history, order=(4, 1, 0))
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


"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:               yield_kg   No. Observations:                  305
Model:                 ARIMA(5, 1, 0)   Log Likelihood                -760.554
Date:                Sun, 13 Sep 2026   AIC                           1533.107
Time:                        18:07:31   BIC                           1555.410
Sample:                    06-14-2004   HQIC                          1542.029
                         - 04-14-2005                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.4406      0.060     -7.406      0.000      -0.557      -0.324
ar.L2         -0.3648      0.057     -6.387      0.000      -0.477      -0.253
ar.L3         -0.1854      0.062     -2.985      0.003      -0.307      -0.064
ar.L4         -0.2373      0.047     -5.020      0.000      -0.330      -0.145
ar.L5         -0.0946      0.057     -1.672      0.095      -0.205       0.016
sigma2         8.7079      0.358     24.353      0.000       8.007       9.409
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):              1856.23
Prob(Q):                              0.93   Prob(JB):                         0.00
Heteroskedasticity (H):               0.33   Skew:                            -1.00
Prob(H) (two-sided):                  0.00   Kurtosis:                        14.94
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
                0
count  305.000000
mean     0.173017
std      3.365679
min    -22.483923
25%     -1.507963
50%      0.132198
75%      1.517422
max     28.175000

ARIMA(5, 1, 0) Test RMSE: 3.097
ARIMA(5, 1, 1) Test RMSE: 3.150
ARIMA(4, 1, 0) Test RMSE: 3.048
ARIMA(4, 1, 1) Test RMSE: 3.141
ARIMA(3, 1, 0) Test RMSE: 3.054
ARIMA(3, 1, 1) Test RMSE: 3.104
"""
