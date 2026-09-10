# ARIMA — Time Series Forecasting

Este projeto é um estudo prático de séries temporais e previsão utilizando o modelo ARIMA (AutoRegressive Integrated Moving Average) em Python.

O projeto utiliza o conjunto de dados **Shampoo Sales**, que contém 36 observações mensais de vendas de shampoo entre janeiro de 1901 e dezembro de 1903. A partir desses dados, são explorados conceitos fundamentais de análise e modelagem de séries temporais.

## Objetivos

O principal objetivo é compreender, de forma prática, como um modelo ARIMA pode ser utilizado para analisar dependências temporais e realizar previsões de valores futuros.

Durante o desenvolvimento, são estudados e implementados conceitos como:

* Séries temporais e sua estrutura temporal;
* Autocorrelação;
* Defasagens (lags);
* Diferenciação de séries temporais;
* Modelos autorregressivos (AR);
* Modelo ARIMA;
* Análise de resíduos;
* Previsão de um passo à frente;
* Walk-forward validation;
* Avaliação de previsões utilizando RMSE.

## Modelo ARIMA

O modelo utilizado inicialmente é um **ARIMA(5, 1, 0)**, em que:

* `p = 5`: utiliza cinco defasagens na componente autorregressiva;
* `d = 1`: aplica uma diferenciação à série para trabalhar com sua variação ao longo do tempo;
* `q = 0`: não utiliza uma componente de média móvel.

O modelo é ajustado utilizando a biblioteca `statsmodels`.

## Autocorrelação

Além da utilização de ferramentas prontas, a autocorrelação é calculada manualmente para compreender como uma série se relaciona com seus próprios valores anteriores.

Para cada lag, são comparados os desvios dos valores em relação à média da série. Isso permite observar a existência de dependência temporal e compreender melhor o comportamento que modelos autorregressivos tentam capturar.

## Análise dos resíduos

Após o ajuste do modelo, seus resíduos são analisados por meio de:

* gráfico dos resíduos;
* estimativa da distribuição por KDE;
* estatísticas descritivas;
* autocorrelação dos resíduos;
* testes estatísticos fornecidos pelo `statsmodels`.

A análise busca verificar se o modelo conseguiu capturar adequadamente a estrutura temporal presente nos dados.

## Walk-Forward Validation

Para avaliar o modelo, os dados são divididos em conjuntos de treinamento e teste.

Em vez de realizar todas as previsões futuras de uma única vez, é utilizada uma estratégia de previsão contínua:

1. O modelo é ajustado utilizando os dados disponíveis.
2. É realizada uma previsão para o próximo instante.
3. O valor real é observado.
4. O novo valor é incorporado ao histórico.
5. O modelo é ajustado novamente.
6. Uma nova previsão é realizada.

Esse processo é repetido até que todas as observações do conjunto de teste tenham sido previstas.

Essa abordagem respeita a ordem temporal dos dados e simula melhor um cenário no qual novas observações chegam progressivamente.

## Avaliação

As previsões são comparadas com os valores reais utilizando o **Root Mean Squared Error (RMSE)**:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
(y_t-\hat{y}_t)^2
}
$$

Quanto menor o RMSE, menor é a magnitude média dos erros de previsão.

## Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Statsmodels
* Scikit-learn

## Estrutura do estudo

O projeto acompanha progressivamente o processo de construção e avaliação de um modelo de previsão:

**dados → análise da série → autocorrelação → ARIMA → resíduos → previsão → walk-forward validation → avaliação**

O objetivo não é apenas obter previsões, mas compreender matematicamente e computacionalmente como os modelos de séries temporais funcionam.
