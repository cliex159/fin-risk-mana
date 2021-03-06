# (PART) Chapter 2 Univariate Volatility {.unnumbered}

# Stationary processes

The volatility plays a crucial role in financial risk management, it is
the main measure of risk. On the other hands, the volatility is the
key factor in, e.g., Investment decisions, Portfolio construction
(Markowitz model) and Derivative pricing (Black-Scholes model).

* In this Chapter we focus on the estimation and forecasting of
volatility for a single asset (univariate).
* The volatility plays a crucial role in financial risk management, it is
the main measure of risk. On the other hands, the volatility is the
key factor in, e.g., Investment decisions, Portfolio construction
(Markowitz model) and Derivative pricing (Black-Scholes model).
* In this Chapter we focus on the estimation and forecasting of
volatility for a single asset (univariate).

## Time series

* A time series is a sequence of observations in chronological order. For example: daily log returns on a stock or monthly values of the Consumer Price Index (CPI).
* A stochastic process is a sequence of random variables and can be viewed as the “theoretical” or “population” analog of a time series, conversely, a time series can be considered a sample from a stochastic process.
* Denote $\{X_t, t \in I\}$ the time series, where I is a time index. For example: $I = \{1, 2, 3, ...\}$ or $I = \{2000, 2001, 2002...2021\}$. Equally spaced time series are the most common in practice. This is the case of
$I = \{t_1, t_2, ..., t_n\}$, where
$(\Delta = t_{i+1} − t)_i$ with $\Delta$ is a constant.

### Remark

**Difference from traditional Statistical Inference**

* In traditional statistic inference, the data is assumed to be an i.i.d
process (random sample).
* In time series, we do not need this assumption and wish to model the dependency among observations which leads to the concept of autocorrelation.

**Some main problems in time series**

* Formulate and estimate a parametric model for $X_t$ (need to propose methods of estimation and model diagnostics).
* This point is related to the estimation of autoregressive (AR) or ARMA models.
* Estimation of Missing values (fill“gaps”).
* Prediction or Forecasting (“would like to know what a future value is”). For example our data is $x_1, x_2, ..., x_{100}$, we wish to forecast the next 10 values, $x_{101}, ..., x_{110}$. In this case, our forecasting horizon is 10.
* Plotting time series to observe fluctuations of time series, e.g., to find stationarity or non-stationarity, cycles, trends, outliers or interventions. Assisting in the formulation of a parametric model.

### Example {.unnumbered}

Consider Financial Index SP500. The data consists of excess returns $X_t = \ln(S_t) −\ln(S_{t−1})$. From the plot we see the following properties of $X_t$:


```r
library(tidyverse)
sp500 = read.csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vT4WqdVoUIiaMcd4jQj5by3Oauc6G4EFq9VDDrpzG2oBn6TFzyNE1yPV2fKRal5F7DmRzCtVa4nSQIw/pub?gid=279168786&single=true&output=csv")

sp500$Close %>% 
  log %>% 
  diff %>% 
  plot(type = "l",col = "blue")
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-1-1.png" width="90%" style="display: block; margin: auto;" />

* The mean level of the process seems constant.
* There are sections of the data with explosive behavior (high volatility).
* The data corresponds to a non-stationary process (will define more detailed).
* The variance (or volatility) is not constant in time.
* No linear time series model will be available for this data.

## Autocovariance

### Definition

The autocovariance function a stochastic process $X$ is defined as 
$$\gamma(t,\tau)=\mathbb{E}(X_t −\mu_t)(X_{t−\tau} −\mu_{t−\tau})$$
for $\tau \in \mathbb{Z}$, where $\mu_t = E(X_t)$.

* The autocovariance function is symmetric, i.e., $\gamma(t,\tau) = \gamma(t − \tau,−\tau)$. For special case $\tau = 0$ then $\gamma(t, 0) = Var(X_t)$.
* In general $\gamma(t,\tau)$ is depend on t as well as $\tau$.

### Example {.unnumbered}

Find the autocovariance function of Brownian motion?

>$$\begin{align*}
&B_t \sim \mathcal{N}(0,t) \\
&\rightarrow E[B_t^2] =Var(B_t)=t
\end{align*}$$

>$$\begin{align*}
&B_t-B_{t-\tau} \sim \mathcal{N}(0,t-\tau) \\
&\rightarrow E[(B_t-B_{t-\tau})^2]=Var(B_t-B_{t-\tau})=t-\tau
\end{align*}$$

>$$\begin{align*}
\gamma(t,\tau)&=E[(B_t-\mu_t)(B_{t-\tau}-\mu_{t-\tau})] \\
&=E[B_tB_{t-\tau}] \\
&=-\frac{1}{2}E[(B_t-B_{t-\tau})^2-B_t^2-B_{t-\tau}^2] \\
&=-\frac{1}{2}\{E[(B_t-B_{t-\tau})^2] -E[B_t^2]-E[B_{t-\tau}^2]\} \\
&=-\frac{1}{2} [(\tau)-(t)-(t-\tau)]=t-\tau
\end{align*}$$

>Answer: The autocovariance function of Brownian motion is $t-\tau$.

##  Stationary

### Strictly Stationary

A process is said to be strictly stationary if all aspects of its behavior are unchanged by shifts in time. Mathematically, stationarity is defined as the requirement that for every $m$ and $n$ the distribution of
$(X_1, X_2, ..., X_n)$ and $(X_{1+m}, X_{2+m}, ..., X_{n+m})$ are the same.

### Weakly Stationary

A process is weakly stationary if its mean, variance, and covariance are unchanged by time shifts. More precisely, $X_1, X_2, ...$ is a weakly stationary process if

1. $\mathbb{E}(X_t)=\mu, \forall t$
2. $Var(X_t) = \sigma_2$ (a positive finite constant) for all $t$.
3. $Cov(X_t, X_s) = \gamma(|t − s|), \forall t, s$ and some function $\gamma$.

We see that, the mean and variance do not change with time and the covariance between two observations depends only on the lag, the time distance $|t − s|$.

* The function $\gamma$ is the autocovariance function of the process and has symmetric property $\gamma(h) = \gamma(−h)$.

$$\begin{align*}
\gamma(h)=cov(X_t,X_{t+h}) \\
\rightarrow \gamma(-h)=cov(X_{t},X_{t-h})
\end{align*}$$

Let $s=t-h$ then $t=s+h$
$$\gamma(-h)=cov(X_{s+h},X_{s})=\gamma(h) $$

* The correlation between $X_t$ and $X_{t+h}$ is denoted by $\rho(h)$. Function $\rho$ is called autocorrelation function (ACF). We have $\gamma(0) = \sigma^2$ and, hence
$\gamma(h) = \sigma^2 \rho(h)$ hence $\rho(h) = \frac{\gamma(h)}{\gamma(0)}$.

The ACF is normalized on $[−1, 1]$. Since the process is required to be covariance stationary, the ACF depends only on one parameter, lag $h$.

### Example {.unnumbered}

Consider the random walk $X: X_t = c + X_{t−1} + \epsilon_t$, with c is constant and white noise $\epsilon_t$. We see that if $c \neq 0$, then $Z_t := X_t −X_{t−1} = c+ \epsilon_t$ have a non-zero mean. We call it a random walk with drift. Note that since $\epsilon_t$ is independent then we call $X_t$ a random walk with independent increments. For more convenience, assume that $c$ and $X_0$ are set to zero. We have 

$$\begin{align*}
&X_t =\epsilon_t + \epsilon_{t−1} +...+\epsilon_1 \\
\\
&\mu_t =E(X_t)=0 \\
\\
&Var(X_t) = t\sigma
\end{align*}$$

$Var(X_t)$ is not stationary but rather increases linearly with time and makes the random walk “wander”, i.e., $X_t$ takes increasingly longer excursions away from its conditional mean of $0$, and therefore is not mean-reverting.

If $s<t$ then

$$ \rho(t,s)=\sqrt{1-\frac{s}{t}} $$

which against $\rho$ depending on $t$ as well as on $s$, thus the random walk is not covariance stationary. The following figure shows the relationship among different processes: Stationary processes are the largest set, followed by white noise, martingale difference (MD), and i.i.d. processes.

<center>
<img src="https://raw.githubusercontent.com/ThanhDatIU/frisk2/main/21.png" alt="" width="50%" height="50%">
</center>

## Estimating Parameters

Let $X_1, X_2, ..., X_n$ be observations from weakly stationary process. To estimate the autocovariance function, we use the sample autocovariance function defined by

$$ \hat{\gamma}(h)=\frac{1}{n} \sum_{t=1}^{n-h}(X_{t+h}-\bar X)(X-t-\bar X) $$

To estimate function $\rho$, we use the sample autocorrelation function
(sample ACF) defined as

$$\hat \rho(h) =\frac{\hat \gamma(h)}{\hat \gamma(h)}$$

* To visualize the dependencies of $x_t$ for different lags h, we use the Correlogram.
* A correlogram is a plot of $h$ (x-axis) versus its corresponding value of $\hat \rho(h)$ (y-axis).
* The correlogram may exhibit patterns and different degrees of dependency in a time series.
* A “band” of size $\frac{2}{\sqrt{n}}$ is added to the correlogram because asymptotically $\hat \rho(h) \sim \mathcal{N} \left(0, \frac{1}{n} \right)$ if the data is close to a white noise process.
* This band is used to detect significant autocorrelations, i.e. autocorelations that are different from zero.


```r
library(tidyquant)
msft = tq_get('MSFT',from=as.Date("2010-01-01"),
               to=as.Date("2014-01-01"),
               get = "stock.prices")

msft_logret=msft$adjusted %>% 
  log() %>% 
  diff()

acf(msft_logret,lag.max=10)
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

## The ADF Test

ADF Test is also called Unit Root Test. The test uses the following null and alternative hypotheses:

* $H_0$ : The time series contains a unit root. This means that the time series is non-stationary, i.e., it has some time-dependent structure and does not have constant variance over time.
* $H_1$ : The time series is stationary.


```r
library(tseries)
adf.test(msft_logret)
```

```
#> 
#> 	Augmented Dickey-Fuller Test
#> 
#> data:  msft_logret
#> Dickey-Fuller = -9.7881, Lag order = 10, p-value = 0.01
#> alternative hypothesis: stationary
```

## KPSS test

The ideas of KPSS test comes from the regression model with time
  trend

$X_t =c+ \mu_t+k \sum_{i=1}^{t} \xi_i +\eta_t$

with stationary $\eta_t$ and i.i.d $\xi$ with mean $0$ and variance $1$. Note that the third term is a random walk. So we set the null hypothesis: the data is stationary and.

$$ H_0 : k = 0 \\
H_1 : k \neq 0 $$

Test results for Microsoft data


```r
library(tseries)
kpss.test(msft_logret)
```

```
#> 
#> 	KPSS Test for Level Stationarity
#> 
#> data:  msft_logret
#> KPSS Level = 0.20346, Truncation lag parameter = 7, p-value = 0.1
```

##  Ljung–Box Test

Sample ACF with test bounds.

* These bounds are used to test the null hypothesis that an autocorrelation coefficient is $0$. 
* The null hypothesis is rejected if the sample autocorrelation is outside the bounds.
* The usual level of the test is $0.05$.

### Example {.unnumbered}

(The First-order Autoregression Model (AR(1))) The time series $X = (X_t)$ is called AR(1) if the value of X at time t is a linear function of the value of $X$ at time $t − 1$ as follows

$$X_t=\delta+\phi_1 X_{t-1}+w_t=\delta+\sum_{h=0}^\infty \phi_1^h w_{t-h} $$

1. The errors $w_t \sim \mathcal{N}(0,\sigma_w^2)$ are i.i.d.
2. $w_t$ is independent of $X_t$.
3. $\phi_1<1$. This condition guarantees that $X_t$ is weakly stationary.

$$\begin{align*} 
&\mu=\mathbb{E}(X_t)=\frac{\delta}{1-\phi_1} \\
\\
&Var(X_t)=\frac{\sigma_w^2}{1-\phi_1^2} \\
\\
&Cov(X_t,X_{t+h})=\gamma(h)=\phi_1^h \cdot \frac{\sigma_w^2}{1-\phi_1^2} \\
\\
&\rho(h)=\phi_1^h
\end{align*}$$

Note that the magnitude of its ACF decays geometrically to zero, either slowly as when $\phi_1 = 0.95$, moderately slowly as when $\phi_1 = 0.75$, or rapidly as when $\phi_1 = 0.25$. We now simulate AR(1) and plot the ACF with $\phi_1 =0.64$ and $\sigma_w^2 =1$.


```r
library(stats)
ts.sim = arima.sim(list(order = c(1,0,0), ar = 0.64), n = 100,sd=1)
plot(ts.sim,col="blue")
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-5-1.png" width="90%" style="display: block; margin: auto;" />

```r
acf(ts.sim)
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-5-2.png" width="90%" style="display: block; margin: auto;" />

The null hypothesis of the Ljung–Box test is
$$H_0 :\rho(1)=\rho(2)=...\rho(m)=0$$
for some m. If the Ljung–Box test rejects, then we conclude that one or more of $\rho(1), \rho(2), ..., \rho(m)$ is nonzero. The Ljung–Box test is sometimes called simply the Box test.
$$Q(m)=n(n+2) \sum_{i=j}^m \frac{\hat p^2 (j)}{n-j} \sim \chi^2(m)$$

### Example {.unnumbered}

Consider AR(1) with $\phi_1 = 0.64$ and $\sigma_w^2 = 1$, we have the results of Box test in R


```r
library(stats)
ts.sim = arima.sim(list(order = c(1,0,0), ar = 0.64), n = 100,sd=1)
plot(ts.sim,col="blue")
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-6-1.png" width="90%" style="display: block; margin: auto;" />

```r
Box.test(ts.sim, lag = 10, type = "Ljung-Box")
```

```
#> 
#> 	Box-Ljung test
#> 
#> data:  ts.sim
#> X-squared = 75.684, df = 10, p-value = 3.5e-12
```

If $|\phi_1| \geq 1$ then AR(1) process is nonstationary, and the mean, variance, covariances and and correlations are not constant.

##  PACF

A partial correlation is a conditional correlation. It is the correlation between two variables under the assumption that we know and take into account the values of some other set of variables.

### Example {.unnumbered}

Consider regression model in which $y$ is the response variable, $x_1, x_2, x_3$ are predictor variables. The partial correlation between y and $x_3$ is the correlation between the variables determined taking into account how both $y$ and $x_3$ are related to $x_1$ and $x_2$.

In regression, this partial correlation could be found by correlating the residuals from two different regressions:

1. Regression in which we predict $y$ from $x_1$ and $x_2$. 
2. Regression in which we predict $x_3$ from $x_1$ and $x_2$.

We correlate the “parts” of $y$ and $x_3$ that are not predicted by $x_1$ and $x_2$. We can define the partial correlation just described as
$$\frac{Cov(y, x_3 | x_1, x_2)}{\sqrt{Var(y | x_1, x_2)Var(x_3 | x_1, x_2)}}$$

For a time series, the partial autocorrelation between $x_t$ and $x_{t−h}$ is defined as the conditional correlation between $x_t$ and $x_{t−h}$ conditional on $x_{t−h+1}, ..., x_{t−1}$, the set of observations that come between the time points $t$ and $t − h$.

$$\frac{Cov(y, x_3 | x_1, x_2)}{\sqrt{Var(y | x_1, x_2)Var(x_3 | x_1, x_2)}}$$

### Example {.unnumbered}

The 3rd order (lag) partial autocorrelation is:

$$\frac{Cov(x_t, x_{t-3} | x_{t-1}, x_{t-2})}{\sqrt{Var(x_t | x_t, x_{t-3})Var(x_{t-3} | x_t, x_{t-3})}}$$

# EWMA

Denote $y_t$ the return of stock at time $t$. Then

* Volatility a weighted sum of past returns, with weights $\omega_i$, is
defined by
$$ \hat \sigma_t^2=\omega_1y_{t-1}^2+\omega_2y_{t-2}^2+...+\omega_Ly_{t-L}^2 $$
where L is the length of the estimation window, i.e., the number of observations used in the calculation. This is called MA model.
* An extension of MA model is Exponentially weighted moving average. Let the weights be exponentially declining, and denote them by $\lambda^i$
$$ \hat \sigma_t^2=\lambda y_{t-1}^2+\lambda^2 y_{t-2}^2+...+\lambda^L y_{t-L}^2 $$
where $0 < \lambda < 1$. If $L$ is large enough, the term αn are negligible for all $n > L$. So we set $L = \infty$.
* Note that the sum of weights is
$$\frac{\lambda}{1-\lambda}=\sum_{i=1}^\infty \lambda^i$$
So the exponentially weighted moving average is defined by
$$ \hat \sigma_t^2=\frac{1-\lambda}{\lambda} \sum_{i=1}^{\infty}\lambda^i y_{t-i}^2 $$
and, hence, we get the EWMA equation (why???)
$$ \hat \sigma_t^2=\lambda \hat \sigma_{t-1}^2+(1-\lambda)y_{t-1}^2 $$
* Note that JP Morgan set for daily data with $\lambda = 0.94$.

## Example {.unnumbered}

Suppose that $\lambda = 0.9$, the volatility estimated for a market variable for
day $n − 1$ is $1\%$ per day, and during day $n − 1$ the market variable
increased by $2\%$. This means that $\sigma_{n-1}^2=0.01^2=0.0001$ and $y_{n-1}^2=0.02^2=0.0004$. From equation (1) we get
$$\sigma_n^2=0.9 \cdot 0.0001 + 0.1 \cdot 0.0004=0.00013 $$
The estimate of the volatility for day $n$ is $\sigma_n = \sqrt{0.00013} = 1.4\%$ per
   day. Note that the expected value of $y_{n-1}^2$ is $\sigma_{n-1}^2= 0.0001$. Hence, realized value of $y_{n−1}^2 = 0.0002$ is greater that expected value, and as a
result our volatility estimate increase. If the realized value of $y_{n−1}^2$ has been less than its expected valued, our estimate of the volatility would have decreased.

# ARCH and GARCH

## ARCH

* The ARCH model was proposed by Robert Engle in 1982 called autoregressive conditionally heteroscadastic.
* Most volatility models derive from this.
* Returns are assumed to have conditional distribution (here
assumed to be normal)
$$y_t \sim \mathcal{N} (0,\sigma_t^2)$$
or we can write
$$y_t=\sigma_t \epsilon_t $$
where $\epsilon_t \sim \mathcal{N}(0, 1)$ is called residual.

ARCH(L1) is defined by
$$Var(y_t | y_{t−1}, y_{t−2}, ..., y_{t−L_1} ) = \sigma_t^2 = \omega + \sum_{i=1}^{L_1} \alpha_i y_{t−i}^2$$
where $L_1$ is called the lag of the model. It is seen that in the ARCH model, the volatility is weighted average of past returns.  

The most common form is ARCH (1)
$$Var(y_t | y_{t−1}) = \sigma_t^2 = \omega + \alpha y_{t−1}^2$$
where $\omega$ and $\alpha$ are parameters that can be estimated by maximum likelihood.

If we assume that the series has $mean = 0$ (this can always be done by centering), then the ARCH model could be written as
$$\begin{align*}
&y_t = \sigma_t \epsilon_t \\
&\sigma_t=\sqrt{\omega+\alpha y_{t-1}^2} \\
&\epsilon_t \sim \mathcal{N}(0,1),i.i.d
\end{align*}$$

We require that $\omega,\alpha>0$ so that $\omega+\alpha y_{t-1}^2>0, \forall t$. We also require that $\alpha < 1$ in order to the process to be stationary with a finite variance. Now we have
$$y_t^2 =\epsilon_t^2(\omega+\alpha y_{t−1}^2)$$
which is similar to an AR(1) for variable $y_t^2$ and with multiplicative noise with a mean of $1$ rather than additive noise with a mean of $0$.

## GARCH

* It turns out that ARCH model is not a very good model and almost nobody uses it. Because, it needs to use information from many days before t to calculate volatility on day t. That is, it needs a lot of lags.
* The $GARCH(L_1, L_2)$ model is defines as
$$ \sigma_t^2=\omega+\sum_{i=1}^{L_1} \alpha_i y_{t-i}^2 + \sum_{i=1}^{L_2} \beta_i \sigma_{t-i}^2 $$ 
and, hence, $GARCH(1,1)$
$$ \sigma_t^2=\omega+\alpha y_{t-1}^2+\beta \sigma_{t-1}^2 $$
* $GARCH(1,1)$ is the most common specification.

### Unconditional volatility

* The unconditional volatility (so-called the long-run variance rate) is the unconditional expectation of volatility on given time
$$\sigma^2=\mathbb{E}(\sigma_t^2) $$
so we have
$$ \sigma^2=\mathbb{E}(\omega+\alpha y_{t-1}^2+\beta \sigma_{t-1}^2)=\omega +\alpha \sigma^2+\beta \sigma^2 $$
Hence,
$$ \sigma^2=\frac{\omega}{1-\alpha-\beta} $$
* So to ensure positive volatility forecasts we need the condition $\omega, \alpha, \beta \geq 0$
Because if any parameter is negative $\sigma_{t+1}$ may be negative.
* For stationary we need condition $\alpha+\beta<1$
Setting $\gamma := 1 − \alpha − \beta$ and $V := \sigma^2$ (called long-run variance rate). We have
$$ \sigma_t^2=\gamma V+\alpha y_{t-1}^2+\beta \sigma_{t-1}^2 $$

### Meaning of Parameters

* The parameter $\alpha$ is news, it shows that how the volatility reacts to new information.
* The parameter $\beta$ is memory, it shows that how much volatility remembers from the past.
* The sum $\alpha + \beta$ determines how quickly the predictability (memory) of the process dies out:
<ul>
<li> if $\alpha + \beta \approx 0$ predictability will die out very quickly.</li> 
<li> if $\alpha + \beta \approx 1$ predictability will die out very slowly.</li>
</ul>

#### Example {.unnumbered}

Suppose that a $GARCH(1,1)$ model is estimated from daily data is 
$$\sigma_n =0.000002+0.13y_{n-1}^2 +0.86 \sigma_{n-1}^2$$
This corresponds to $\omega = 0.000002, \alpha = 0.13, \beta = 0.86$. We have
$$\sigma^2 = \frac{\omega}{1-\alpha-\beta}= 0.0002$$
or $\sigma=\sqrt{0.0002}=0.014=1.4\%$ per day.  

Suppose that the estimate of the volatility on day $n − 1$ is $1.6\%$ per day
so that $\sigma^2 = 0.0162 = 0.000256$, and on that day $n − 1$ the market
variable decreased by $1\%$ so that $y_{n−1}^2 = 0.01^2 = 0.0001$. Then
$$\sigma_n^2 = 0.000002 + 0.13 × 0.0001 + 0.86 × 0.000256 = 0.00023516$$
the new estimate of the volatility is: $\sqrt{0.00023516} = 0.0153$ or $1.53\%$ per day.

# Maximum likelihood

Maximum likelihood is the most important and widespread method of estimation. What is maximum likelihood?

Ask the question which parameters most likely generated the data we have. Suppose we have a sample of $\{−0.2, 3, 4, −1, 0.5\}$. In the following three possibilities, which is most likely for parameters?


| case | $μ$ | $σ$ |
|------|:-----:|---------:|
| 1 | 1 | 5 |
| 2 | -2 | 2 |
| 3 | 1 | 2 |

Let $Y = (y_1,y_2,...,y_n)$ be a vector of data and let $\theta = (\theta_1,\theta_2,...,\theta_p)$ be a vector of parameters. Let $f(Y | \theta)$ be the density of Y which depends on the parameters. The function
$$L(\theta) := f(Y | \theta)$$
is viewed as the function of $\theta$ with $Y$ fixed at the observed data is called
the likelihood function.

* The maximum likelihood estimator (MLE) is the value of $\theta$ that maximizes the likelihood function. We denote the MLE by $\hat \theta_{ML}$.
* It is mathematically easier to maximize $\ln L(\theta)$, which is called the log-likelihood. If the data are independent, then the likelihood is the product of the marginal densities.

## Application to ARCH(1)

Consider ARCH(1) model:

$$\epsilon_t ∼ N(0,1)$$

For $t = 2$ we have the density??

$$ f(y_2|y_1)=\frac{1}{\sqrt{2\pi(\omega+\alpha y_1^2)}} e^{-\frac{1}{2} \frac{y_2^2}{2\omega+\alpha y_1^2}} $$
Hence, the joint density
$$ \prod_{t=2}^T f(y_t|y_{t-1})=\prod_{t=2}^T \frac{1}{\sqrt{2\pi(\omega+\alpha y_{t-1}^2)}} e^{-\frac{1}{2} \frac{y_t^2}{2\omega+\alpha y_{t-1}^2}} $$
and, the log likelihood
$$ \ln(L(\omega, \alpha)) =-\frac{T-1}{2} \ln(2\pi)-\frac{1}{2} \sum_{t=2}^T \left( \ln(\omega+\alpha y_{t-1}^2) + \frac{y_t^2}{\omega+\alpha y_{t-1}^2} \right) $$

##  Application to GARCH(1,1)

$$ \sigma_t^2=\omega+\alpha y_{t-1}^2 +\beta \sigma_{t-1}^2$$
Hence, the joint density
$$ f(y_2|y_1)=\frac{1}{\sqrt{2\pi(\omega+\alpha y_1^2+\beta \hat \sigma_1^2)}} e^{-\frac{1}{2} \frac{y_2^2}{\omega +\alpha y_1^2+\beta \hat \sigma_1^2}} $$
and, the log likelihood
$$ \ln(L(\omega, \alpha)) =-\frac{T-1}{2} \ln(2\pi)-\frac{1}{2} \sum_{t=2}^T \left( \ln(\omega+\alpha y_{t-1}^2+\beta \hat \sigma_{t-1}^2) + \frac{y_t^2}{\omega+\alpha y_{t-1}^2+\beta \hat \sigma_{t-1}^2} \right) $$

### The importance of σ1

* $\sigma_1$ can make a large difference.
* Especially when the sample size is small. 
* Typically set $\sigma_1 = \hat \sigma$.

### Volatility targeting {.unnumbered}

* Since we have the long-run variance rate 
$$\sigma^2= \frac{\omega}{1-\alpha-\beta}$$.
* We can set
$$ \omega=\hat \sigma^2(1-\alpha-\beta) $$
where $\hat \sigma^2$ is is the sample variance.
* Hence we save one parameter in the estimation.

# Future volatility

The variance rate estimated at the end of day $n − 1$ for $n$ day when apply $GARCH(1,1)$ model is

$$ \sigma_n^2=\omega+\alpha y_{n-1}^2+\beta \sigma_{n-1}^2=\sigma^2(1-\alpha-\beta)+\alpha y_{n-1}^2+\beta \sigma_{n-1}^2 $$
or
$$ \sigma_n^2- \sigma^2=\alpha (y_{n-1}^2-\sigma^2)+\beta (\sigma_{n-1}^2-\sigma^2) $$
On day $n+t$ in the future we have
$$ \sigma_{n+t}^2 - \sigma^2=\alpha (y_{n+t-1}^2-\sigma^2)+\beta(\sigma_{n+t-1}^2-\sigma^2) $$
Hence,
$$ \mathbb{E}[\sigma_{n+t}^2 - \sigma^2]=(\alpha+\beta)\mathbb{E}[\sigma_{n+t-1}^2 - \sigma^2] $$
By induction we obtain
$$ \mathbb{E}(\sigma_{n+t}^2) = \sigma^2 + (\alpha + \beta)^t(\sigma_n^2 − \sigma^2) $$

## Example {.unnumbered}

For the S&P data consider earlier, $\alpha + \beta = 0.9935$, the log-run variance rate $\sigma^2 = 0.0002075$ (or $\sigma = 1.44\%$ per day). Suppose that our estimate of the current variance rate per day is $0.0003$ (This corresponds to a volatility of $1.732\%$ per day). In $t = 10$ days, calculate the expected variance rate??  

We have $\sigma_n^2 = 0.0003$, hence
$$\begin{align*}
\mathbb{E}(\sigma_{n+10}^2 ) = 0.0002075 + 0.993510^{10} × (0.0003 − 0.0002075) = 0.0002942
\end{align*}$$
or the expected volatility per day is $\sqrt{0.0002942} = 1.72\%$, still above the long-term volatility of $1.44\%$ per day.

##  Volatility term structures {.unnumbered}

Suppose it is day $n$. We define
$$V(t) = \mathbb{E}(\sigma_{n+1}^2 )$$
and
$$ a:= \ln \left( \frac{1}{\alpha+\beta} \right) $$
From $\mathbb{E}(\sigma_{n+t}^2) = \sigma^22 + (\alpha + \beta)^t(\sigma_n^2 − \sigma^2)$ we have

$$V(t) = \sigma^2 + e^{−at}(V(0) − \sigma^2)$$
Then we have the average variance rate per day between today and time T.

$$ \frac{1}{T} \int_0^T V(t) \,dt=\frac{1}{T} \int_0^T \left( \sigma^2 + e^{−at}(V(0) − \sigma^2) \right)=\sigma^2+\frac{1-e^{-aT}}{aT} [V(0)-\sigma^2] $$

Now we define $sigma(T)$ the volatility per annum that should be used to price a T-day option under $GARCH(1,1)$ model. Then we have

$$ \sigma^2(T)=252 \left( \sigma^2+\frac{1-e^{-aT}}{aT} [V(0)-\sigma^2] \right) $$
This relationship between the volatility of options and their maturities is referred to as the volatility term structure.

### Example {.unnumbered}

For S&P data, using GARCH(1,1) model we obtain the coefficients $\omega = 0.0000013465$, $\alpha = 0.083394$ and $\beta = b = 0.910116$. So from 
$$ \sigma^2(T)=252 \left( \sigma^2+\frac{1-e^{-aT}}{aT} [V(0)-\sigma^2] \right) $$
assume that $V(0) = 0.0003$ we have
$$ \sigma^2=\frac{0.0000013465}{1 − 0.083394 − 0.910116}=0.0002073 $$
and $a = \ln \left( \frac{1}{0.99351} \right) = 0.00651$. Hence,
$$ \sigma^2(T)=252 \left( 0.0002073+\frac{1-e^{-0.00651 \cdot T}}{0.00651 \cdot T}[0.0003-0.0002073] \right) $$

For the option life (days) T = 10, 30, 50, 100, 500, we obtain the option
volatility ($\%$ per annum)


| Option life (days) | 10 | 30 | 50 | 100 | 500 |
|------|:-----:|------:|------:|------:|------:|
| Option volatility | 27.36 | 27.10 | 26.87 | 26.35 | 24.32 |

# In-class exercise

## Autocovariance

1. Find the autocovariance function of Brownian motion?

>$$\begin{align*}
&B_t \sim \mathcal{N}(0,t) \\
&\rightarrow E[B_t^2] =Var(B_t)=t
\end{align*}$$

>$$\begin{align*}
&B_t-B_{t-\tau} \sim \mathcal{N}(0,t-\tau) \\
&\rightarrow E[(B_t-B_{t-\tau})^2]=Var(B_t-B_{t-\tau})=t-\tau
\end{align*}$$

>$$\begin{align*}
\gamma(t,\tau)&=E[(B_t-\mu_t)(B_{t-\tau}-\mu_{t-\tau})] \\
&=E[B_tB_{t-\tau}] \\
&=-\frac{1}{2}E[(B_t-B_{t-\tau})^2-B_t^2-B_{t-\tau}^2] \\
&=-\frac{1}{2}\{E[(B_t-B_{t-\tau})^2] -E[B_t^2]-E[B_{t-\tau}^2]\} \\
&=-\frac{1}{2} [(\tau)-(t)-(t-\tau)]=t-\tau
\end{align*}$$

>Answer: The autocovariance function of Brownian motion is $t-\tau$

2. Let $cov (X_t,X_{t+h})=\gamma(h)$

### Question a

Prove that $\gamma(h)=\gamma(-h)$

>$$\begin{align*}
\gamma(h)=cov(X_t,X_{t+h}) \\
\rightarrow \gamma(-h)=cov(X_{t},X_{t-h})
\end{align*}$$

>Let $s=t-h$ then $t=s+h$
$$\gamma(-h)=cov(X_{s+h},X_{s})=\gamma(h) $$

### Question b

Prove that $-1 \leq \rho(h) \le1$

>$$\begin{align*}
&\mathbb{E}[(X_{t+h} \pm X_{t})^2] \ge 0 \\ 
&\rightarrow \mathbb{E}[X_{t+h}^2] + \mathbb{E}[X_{t}^2] \pm 2 \mathbb{E}[X_{t+h}X_{t}] \ge 0 \\ 
&\rightarrow 2 \gamma(0) \pm 2\gamma(h) \ge 0 \\ 
&\rightarrow -2 \gamma(0) \leq 2\gamma(h) \leq 2 \gamma(0) \\
&\rightarrow -1 \leq \rho(h) \leq 1
\end{align*}$$

## ADF test

### Python {.unnumbered} 


```python
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt

price = web.get_data_yahoo("^gspc",
start = "2009-01-01",
end = "2021-12-31")

# Log-data
x=np.log(price['Adj Close'])
plt.plot(x,color="black")
plt.show()

# First difference of log-data
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-9-1.png" width="90%" style="display: block; margin: auto;" />

```python
y=np.diff(np.log(price['Adj Close']))
plt.plot(y,color="black")
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-9-2.png" width="90%" style="display: block; margin: auto;" />


```python
from statsmodels.tsa.stattools import adfuller
result=adfuller(x)
print('ADF Statistic: %f' % result[0])
```

```
#> ADF Statistic: -0.541575
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.883645
```

```python
result=adfuller(y)
print('ADF Statistic: %f' % result[0])
```

```
#> ADF Statistic: -12.311760
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.000000
```

### R {.unnumbered} 


```r
library(tseries)
library(zoo)
price = get.hist.quote(instrument = "^gspc",start = "2009-01-01",
                       end = ("2021-12-31"),  quote = "AdjClose")
```

```
#> time series starts 2009-01-02
#> time series ends   2021-12-30
```

```r
x=coredata(log(price))
y=coredata(diff(log(price)))
# Log-data
ts.plot(x,xlab="time",ylab="returns")
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-11-5.png" width="90%" style="display: block; margin: auto;" />

```r
# First difference of log-data
ts.plot(y,xlab="time",ylab="returns")
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-11-6.png" width="90%" style="display: block; margin: auto;" />


```r
library(aTSA)
adf.test(x)
```

```
#> Augmented Dickey-Fuller Test 
#> alternative: stationary 
#>  
#> Type 1: no drift no trend 
#>       lag  ADF p.value
#>  [1,]   0 2.47    0.99
#>  [2,]   1 2.87    0.99
#>  [3,]   2 2.66    0.99
#>  [4,]   3 2.73    0.99
#>  [5,]   4 2.81    0.99
#>  [6,]   5 2.83    0.99
#>  [7,]   6 3.09    0.99
#>  [8,]   7 2.84    0.99
#>  [9,]   8 3.06    0.99
#> Type 2: with drift no trend 
#>       lag    ADF p.value
#>  [1,]   0 -0.367   0.910
#>  [2,]   1 -0.216   0.930
#>  [3,]   2 -0.275   0.922
#>  [4,]   3 -0.346   0.913
#>  [5,]   4 -0.317   0.917
#>  [6,]   5 -0.374   0.909
#>  [7,]   6 -0.369   0.910
#>  [8,]   7 -0.461   0.893
#>  [9,]   8 -0.494   0.881
#> Type 3: with drift and trend 
#>       lag   ADF p.value
#>  [1,]   0 -4.15  0.0100
#>  [2,]   1 -3.53  0.0391
#>  [3,]   2 -3.82  0.0174
#>  [4,]   3 -3.81  0.0180
#>  [5,]   4 -3.69  0.0241
#>  [6,]   5 -3.72  0.0223
#>  [7,]   6 -3.45  0.0471
#>  [8,]   7 -3.82  0.0177
#>  [9,]   8 -3.63  0.0293
#> ---- 
#> Note: in fact, p.value = 0.01 means p.value <= 0.01
```

```r
adf.test(y)
```

```
#> Augmented Dickey-Fuller Test 
#> alternative: stationary 
#>  
#> Type 1: no drift no trend 
#>       lag   ADF p.value
#>  [1,]   0 -65.9    0.01
#>  [2,]   1 -40.3    0.01
#>  [3,]   2 -33.1    0.01
#>  [4,]   3 -29.5    0.01
#>  [5,]   4 -26.1    0.01
#>  [6,]   5 -25.6    0.01
#>  [7,]   6 -21.3    0.01
#>  [8,]   7 -21.2    0.01
#>  [9,]   8 -19.2    0.01
#> Type 2: with drift no trend 
#>       lag   ADF p.value
#>  [1,]   0 -66.1    0.01
#>  [2,]   1 -40.4    0.01
#>  [3,]   2 -33.3    0.01
#>  [4,]   3 -29.6    0.01
#>  [5,]   4 -26.3    0.01
#>  [6,]   5 -25.8    0.01
#>  [7,]   6 -21.5    0.01
#>  [8,]   7 -21.5    0.01
#>  [9,]   8 -19.4    0.01
#> Type 3: with drift and trend 
#>       lag   ADF p.value
#>  [1,]   0 -66.1    0.01
#>  [2,]   1 -40.4    0.01
#>  [3,]   2 -33.3    0.01
#>  [4,]   3 -29.6    0.01
#>  [5,]   4 -26.3    0.01
#>  [6,]   5 -25.8    0.01
#>  [7,]   6 -21.5    0.01
#>  [8,]   7 -21.5    0.01
#>  [9,]   8 -19.4    0.01
#> ---- 
#> Note: in fact, p.value = 0.01 means p.value <= 0.01
```

## KPSS test

### Python {.unnumbered} 


```python
import statsmodels.api as sm
#perform KPSS test
result=sm.tsa.stattools.kpss(x)
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/stattools.py:2018: InterpolationWarning: The test statistic is outside of the range of p-values available in the
#> look-up table. The actual p-value is smaller than the p-value returned.
#> 
#>   warnings.warn(
```

```python
print('KPSS Statistic: %f' % result[0])
```

```
#> KPSS Statistic: 8.352647
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.010000
```

```python
result=sm.tsa.stattools.kpss(y)
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/stattools.py:2022: InterpolationWarning: The test statistic is outside of the range of p-values available in the
#> look-up table. The actual p-value is greater than the p-value returned.
#> 
#>   warnings.warn(
```

```python
print('KPSS Statistic: %f' % result[0])
```

```
#> KPSS Statistic: 0.036716
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.100000
```

### R {.unnumbered} 


```r
library(tseries)
kpss.test(x)
```

```
#> KPSS Unit Root Test 
#> alternative: nonstationary 
#>  
#> Type 1: no drift no trend 
#>  lag   stat p.value
#>   13 0.0259     0.1
#> ----- 
#>  Type 2: with drift no trend 
#>  lag   stat p.value
#>   13 0.0781     0.1
#> ----- 
#>  Type 1: with drift and trend 
#>  lag  stat p.value
#>   13 0.112     0.1
#> ----------- 
#> Note: p.value = 0.01 means p.value <= 0.01 
#>     : p.value = 0.10 means p.value >= 0.10
```

```r
kpss.test(y)
```

```
#> KPSS Unit Root Test 
#> alternative: nonstationary 
#>  
#> Type 1: no drift no trend 
#>  lag stat p.value
#>   13 2.16  0.0239
#> ----- 
#>  Type 2: with drift no trend 
#>  lag   stat p.value
#>   13 0.0371     0.1
#> ----- 
#>  Type 1: with drift and trend 
#>  lag   stat p.value
#>   13 0.0283     0.1
#> ----------- 
#> Note: p.value = 0.01 means p.value <= 0.01 
#>     : p.value = 0.10 means p.value >= 0.10
```

## Fit ARIMA 

### Python {.unnumbered} 


```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(x, order=(1,0,0))
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
```

```python
model_fit = model.fit()
print(model_fit.summary())
```

```
#>                                SARIMAX Results                                
#> ==============================================================================
#> Dep. Variable:              Adj Close   No. Observations:                 3274
#> Model:                 ARIMA(1, 0, 0)   Log Likelihood                9977.250
#> Date:                Wed, 16 Mar 2022   AIC                         -19948.500
#> Time:                        22:19:50   BIC                         -19930.218
#> Sample:                             0   HQIC                        -19941.953
#>                                - 3274                                         
#> Covariance Type:                  opg                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> const          7.5866      0.498     15.229      0.000       6.610       8.563
#> ar.L1          0.9998      0.000   2474.239      0.000       0.999       1.001
#> sigma2         0.0001   1.21e-06    108.789      0.000       0.000       0.000
#> ===================================================================================
#> Ljung-Box (L1) (Q):                  68.56   Jarque-Bera (JB):             23678.78
#> Prob(Q):                              0.00   Prob(JB):                         0.00
#> Heteroskedasticity (H):               1.01   Skew:                            -0.68
#> Prob(H) (two-sided):                  0.86   Kurtosis:                        16.10
#> ===================================================================================
#> 
#> Warnings:
#> [1] Covariance matrix calculated using the outer product of gradients (complex-step).
```

```python
model = ARIMA(y, order=(1,0,0))
model_fit = model.fit()
print(model_fit.summary())
```

```
#>                                SARIMAX Results                                
#> ==============================================================================
#> Dep. Variable:                      y   No. Observations:                 3273
#> Model:                 ARIMA(1, 0, 0)   Log Likelihood               10015.913
#> Date:                Wed, 16 Mar 2022   AIC                         -20025.825
#> Time:                        22:19:50   BIC                         -20007.545
#> Sample:                             0   HQIC                        -20019.279
#>                                - 3273                                         
#> Covariance Type:                  opg                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> const          0.0005      0.000      2.773      0.006       0.000       0.001
#> ar.L1         -0.1437      0.007    -19.320      0.000      -0.158      -0.129
#> sigma2         0.0001   1.32e-06     97.279      0.000       0.000       0.000
#> ===================================================================================
#> Ljung-Box (L1) (Q):                   0.31   Jarque-Bera (JB):             18498.33
#> Prob(Q):                              0.58   Prob(JB):                         0.00
#> Heteroskedasticity (H):               0.97   Skew:                            -0.83
#> Prob(H) (two-sided):                  0.57   Kurtosis:                        14.53
#> ===================================================================================
#> 
#> Warnings:
#> [1] Covariance matrix calculated using the outer product of gradients (complex-step).
```

### R {.unnumbered} 


```r
fitAR1=arima(x, order = c(1,0,0))
print(fitAR1)
```

```
#> 
#> Call:
#> arima(x = x, order = c(1, 0, 0))
#> 
#> Coefficients:
#>          ar1  intercept
#>       0.9998     7.5866
#> s.e.  0.0002     0.5485
#> 
#> sigma^2 estimated as 0.0001315:  log likelihood = 9973.97,  aic = -19941.94
```

```r
fitAR2=arima(y, order = c(1,0,0))
print(fitAR2)
```

```
#> 
#> Call:
#> arima(x = y, order = c(1, 0, 0))
#> 
#> Coefficients:
#>           ar1  intercept
#>       -0.1437      5e-04
#> s.e.   0.0173      2e-04
#> 
#> sigma^2 estimated as 0.0001285:  log likelihood = 10012.31,  aic = -20018.63
```

## What happens with price

### Python {.unnumbered} 


```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
#what happens if we fit AR for price, not that price is non-stationary
fitAR=ARIMA(price['Adj Close'], order=(1,0,0))
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
```

```python
print(fitAR)
```

```
#> <statsmodels.tsa.arima.model.ARIMA object at 0x7fa897f964c0>
```

```python
result=adfuller(price['Adj Close'])
print('ADF Statistic: %f' % result[0])
```

```
#> ADF Statistic: 1.855653
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.998453
```

```python
result=sm.tsa.stattools.kpss(price['Adj Close'])
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/stattools.py:2018: InterpolationWarning: The test statistic is outside of the range of p-values available in the
#> look-up table. The actual p-value is smaller than the p-value returned.
#> 
#>   warnings.warn(
```

```python
print('KPSS Statistic: %f' % result[0])
```

```
#> KPSS Statistic: 7.902349
```

```python
print('p-value: %f' % result[1])
```

```
#> p-value: 0.010000
```

### R {.unnumbered} 


```r
library(tseries)
library(zoo)
#what happens if we fit AR for price, not that price is non-stationary
fitAR=arima(price$Adjusted, order = c(1,0,0))
print(fitAR)
```

```
#> 
#> Call:
#> arima(x = price$Adjusted, order = c(1, 0, 0))
#> 
#> Coefficients:
#>       ar1  intercept
#>         1   2159.379
#> s.e.    0        NaN
#> 
#> sigma^2 estimated as 513.1:  log likelihood = -15248,  aic = 30501.99
```

```r
adf.test(coredata(price))
```

```
#> Augmented Dickey-Fuller Test 
#> alternative: stationary 
#>  
#> Type 1: no drift no trend 
#>       lag  ADF p.value
#>  [1,]   0 2.96    0.99
#>  [2,]   1 3.59    0.99
#>  [3,]   2 3.25    0.99
#>  [4,]   3 3.19    0.99
#>  [5,]   4 3.46    0.99
#>  [6,]   5 3.42    0.99
#>  [7,]   6 3.77    0.99
#>  [8,]   7 3.28    0.99
#>  [9,]   8 3.62    0.99
#> Type 2: with drift no trend 
#>       lag  ADF p.value
#>  [1,]   0 1.25    0.99
#>  [2,]   1 1.73    0.99
#>  [3,]   2 1.48    0.99
#>  [4,]   3 1.41    0.99
#>  [5,]   4 1.62    0.99
#>  [6,]   5 1.57    0.99
#>  [7,]   6 1.80    0.99
#>  [8,]   7 1.46    0.99
#>  [9,]   8 1.67    0.99
#> Type 3: with drift and trend 
#>       lag    ADF p.value
#>  [1,]   0 -1.238   0.900
#>  [2,]   1 -0.564   0.979
#>  [3,]   2 -0.906   0.952
#>  [4,]   3 -0.951   0.947
#>  [5,]   4 -0.667   0.974
#>  [6,]   5 -0.691   0.971
#>  [7,]   6 -0.351   0.989
#>  [8,]   7 -0.794   0.962
#>  [9,]   8 -0.452   0.984
#> ---- 
#> Note: in fact, p.value = 0.01 means p.value <= 0.01
```

```r
kpss.test(coredata(price))
```

```
#> KPSS Unit Root Test 
#> alternative: nonstationary 
#>  
#> Type 1: no drift no trend 
#>  lag   stat p.value
#>   13 0.0872     0.1
#> ----- 
#>  Type 2: with drift no trend 
#>  lag  stat p.value
#>   13 0.066     0.1
#> ----- 
#>  Type 1: with drift and trend 
#>  lag  stat p.value
#>   13 0.151  0.0458
#> ----------- 
#> Note: p.value = 0.01 means p.value <= 0.01 
#>     : p.value = 0.10 means p.value >= 0.10
```

# Homework {.unnumbered}

## Problem 1 {.unnumbered} 

Give the data


| Time  | Data  | Time  | Data  | Time | Data |
|------:|:-----:|------:|------:|-----:|-----:|
| 1 | 10.31778 | 7 | 10.36644 | 13 | 10.34145 |
| 2 | 10.29235 | 8 | 10.36744 | 14 | 10.40247 |
| 3 | 10.30075 | 9 | 10.39553 | 15 | 10.39158 |
| 4 | 10.29208 | 10 | 10.48562 | 16 | 10.35517 |
| 5 | 10.31304 | 11 | 10.47619 | 17 | 10.35166 |
| 6 | 10.32042 | 12 | 10.46396 | 18 | 10.36395 |

Calculate the ACF with lags from 0 to 15 and lower, upper bounds with significant α = 5%

### Python {.unnumbered}  


```python
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import Data
data = [10.31778, 10.29235, 10.30075, 10.29208, 10.31304, 10.32042,
        10.36644, 10.36744, 10.39553, 10.48562, 10.47619, 10.46396,
        10.34145, 10.40247, 10.39158, 10.35517, 10.35166, 10.36395]

import pandas as pd
df = pd.DataFrame(data, columns = ['data'])
df.head(5)
```

```
#>        data
#> 0  10.31778
#> 1  10.29235
#> 2  10.30075
#> 3  10.29208
#> 4  10.31304
```

```python
from statsmodels.tsa.stattools import acf
# Calculate ACF
print('Calculated ACF by lags 0 - 15:\n')
```

```
#> Calculated ACF by lags 0 - 15:
```

```python
acf(df['data'], nlags = 15)
```

```
#> array([ 1.        ,  0.71637405,  0.48086057,  0.24512236,  0.07189534,
#>        -0.16133272, -0.38073392, -0.39587896, -0.41465118, -0.34811093,
#>        -0.23179094, -0.09359336, -0.02129721, -0.02593246,  0.0128132 ,
#>         0.02943992])
```

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['data'], lags = 15)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-20-1.png" width="90%" style="display: block; margin: auto;" />

### R {.unnumbered}  


```r
# Import Data
data = c(10.31778, 10.29235, 10.30075, 10.29208, 10.31304, 10.32042,
        10.36644, 10.36744, 10.39553, 10.48562, 10.47619, 10.46396,
        10.34145, 10.40247, 10.39158, 10.35517, 10.35166, 10.36395)
```


```r
head(data)
```

```
#> [1] 10.31778 10.29235 10.30075 10.29208 10.31304 10.32042
```


```r
# Calculate ACF
acf(data,lag.max = 15,plot=F)
```

```
#> 
#> Autocorrelations of series 'data', by lag
#> 
#>      0      1      2      3      4      5      6      7      8      9     10 
#>  1.000  0.716  0.481  0.245  0.072 -0.161 -0.381 -0.396 -0.415 -0.348 -0.232 
#>     11     12     13     14     15 
#> -0.094 -0.021 -0.026  0.013  0.029
```


```r
# Plot ACF
acf(data,lag.max = 15)
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-24-1.png" width="90%" style="display: block; margin: auto;" />

## Problem 2 {.unnumbered} 

Consider the $MA(q)$. 

### Question a {.unnumbered} 

Calculate autocovariance $\gamma(h)$ and ACF $\rho(h)$.

>Note that MA(q) is stationary since it is a linear combination of stationary
processes. Note that, white noise has
$$\gamma(0)=Cov(\epsilon_t,\epsilon_{t})=\sigma^2 $$ 
and $\gamma(h)=0$ if $h \neq 0$. For more simple we assume that $E[X_t] = 0$. So we have the
autocovariance is

>$$\begin{align*}
\gamma(h) &= Cov(X_t,X_{t+h}) \\
&=Cov(\sum_{i=1}^q \theta_i \epsilon_{t-i},\sum_{j=1}^q \theta_j \epsilon_{t+h-j}) \\
&= \sum_{i=1}^q \sum_{j=1}^q \theta_i \theta_j Cov(\epsilon_{t-i},\epsilon_{t+h-j}) \\
&= \sum_{i=0}^{q-|h|} \theta_i \theta_{i+|h|}\sigma^2, |h| \leq q
\end{align*}$$
Hence,
$$ \rho(h) = \frac{\sum_{i=0}^{q-|h|} \theta_i \theta_{i+|h|}\sigma^2}{\sum_{i=0}^{q}\theta_i^2} $$ and $\rho(h)=0$ for $|h| > q$

### Question b {.unnumbered} 
  
Show that the MA(1) processes $X_t$ and $Y_t$
$$\begin{align*}
X_t = \beta \epsilon_{t−1} + \epsilon_t \\ 
Y_t = \frac{1}{\beta} \epsilon_{t−1} + \epsilon_t \\ 
\end{align*}$$
have the same ACF.

>Consider the $MA(1)$ model $X_t=\beta \epsilon_{t-1}+\epsilon_t$. The coefficient $\theta_1=\beta$. The ACF is given by
$$ \rho_1=\frac{\beta}{1+\beta^2}, (1) $$
and $\rho_h=0, \forall h \geq 2$ 

>Consider the $MA(1)$ model $Y_t=\frac{1}{\beta} \epsilon_{t-1}+\epsilon_t$. The coefficient $\theta_1=\frac{1}{\beta}$. The ACF is given by
$$\begin{align*} \rho_1&=\frac{\frac{1}{\beta}}{1+\left(\frac{1}{\beta}\right)^2} \\
&=\frac{1}{\beta} \cdot \frac{\beta^2}{1+\beta^2} \\
&=\frac{\beta}{1+\beta^2}, (2) 
\end{align*}$$
and $\rho_h=0, \forall h \geq 2$ 

>(1),(2) imply two $MA(1)$ processes above have the same ACF.

>Answer: $X_t$ and $Y_t$ have the same ACF.

## Problem 3 {.unnumbered} 

Suppose that the stationary process $X_t$ has an autocovariance function given by $\gamma(h)$.
Consider process $Y_t = X_t − X_{t−1}$.

a) Is the process $Y_t$ stationary

>
$$\begin{align*}
\mathbb{E}(Y_t)&=\mathbb{E}(X_t)-\mathbb{E}(X_{t-1}) \\
&=0 \\
Var(Y_t) &=Var(X_t-X_{t=1}) \\
&=Var(X_t)+Var(X_{t-1})-2Cov(X_t,X_{t-1}) \\
&=2\sigma_x^2-2\gamma_x(1) < \infty \\
Cov(Y_t,Y_{t+h})&=Cov(X_t-X_{t-1},X_{t+h}-X_{t+h-1}) \\
&=Cov(X_t,X_{t+h}) 
-Cov(X_t,X_{t+h-1}) 
-Cov(X_{t-1},X_{t+h})
+Cov(X_{t-1},X_{t+h-1}) \\
&=\gamma(h)-\gamma(h-1)-\gamma(h+1)+\gamma(h) \\
&=2\gamma(h)-\gamma(h-1)-\gamma(h+1)
\end{align*}$$

>Answer: $\{Yt\}$ is a weakly stationary proces

b) Find the autocorrelation function of $Y_t$

>
$$\begin{align*}
\rho(h)&=\rho_h(Y_t) \\
&=\frac{\gamma(h)}{\gamma(0)} \\
&=\frac{2\gamma(h)-\gamma(h-1)-\gamma(h+1)}{2 (\gamma(0)-\gamma(1))} 
\end{align*}$$

>Answer: The autocorrelation function of $Y_t$ is $\frac{2\gamma(h)-\gamma(h-1)-\gamma(h+1)}{2 (\gamma(0)-\gamma(1))}$

## Problem 4 {.unnumbered} 

Find the autocorrelation function of the second order moving average process $MA(2)$
$$X_t = 0.5 \epsilon_{t−1} − 0.2 \epsilon_{t−2} + \epsilon_t$$
where $\epsilon_t$ is the white noise.

>Apply problem $2$ with $q=2$, we have
$$ \gamma(h)=\sum_{i=0}^{2-|h|}\theta_i \theta_{i+|h|}\sigma^2, |h| \leq 2 $$
Hence,
$$\begin{align*}
\gamma(0) &= (\theta_0^2 + \theta_1^2 + \theta_2^2) \sigma^2 \\
&=(0.5^2 + 0.2^2 +1)\sigma^2 \\
&=1.29 \sigma^2 \\
\gamma(1) &= (\theta_0 \theta_1 + \theta_1 \theta_2) \sigma^2 \\
&=(1 \cdot 0.5   - 0.5 \cdot 0.2 )\sigma^2 \\
&=0.4 \sigma^2 \\
\gamma(2) &= \theta_0 \theta_2 \sigma^2 \\
&=-0.2 \sigma^2 \\
\end{align*}$$

>
$$\begin{align*}
\gamma(0)&=Var(X_t) \\
&=0.5^2 \sigma^2 +0.2^2 \sigma^2+\sigma^2 \\
&=(0.5^2 + 0.2^2 +1)\sigma^2 \\
&=1.29 \sigma^2 \\
\gamma(1) &=Cov(X_t,X_{t-1}) \\
&=Cov(0.5 \epsilon_{t-1} - 0.2 \epsilon_{t-2}+\epsilon_t,0.5 \epsilon_{t-2} -0.2\epsilon_{t-3}+\epsilon_{t-1}) \\
&=0.5 \cdot 1 Var(\epsilon_{t-1})-0.2 \cdot 0.5 Var(\epsilon_{t-2}) \\
&=(0.5 \cdot 1 -0.2 \cdot 0.5)\sigma^2 \\
&=0.4 \sigma^2 \\
\gamma(2) &=Cov(X_t,X_{t-2}) \\
&=Cov(0.5 \epsilon_{t-1} - 0.2 \epsilon_{t-2}+\epsilon_t,0.5 \epsilon_{t-3} -0.2\epsilon_{t-4}+\epsilon_{t-2}) \\
&=-0.2\cdot 1Var(\epsilon_{t-2}) \\
&=-0.2 \sigma^2
\end{align*}$$

>$$\rho(h)=\begin{cases}
    1&  h=0.\\
    0.31&  h=1\\
    -0.155&  h=2 \\
    0&  h \geq 3
  \end{cases}$$
  
>Answer: The autocorrelation function of the second order moving average process is
$$\rho(h)=\begin{cases}
    1&  h=0.\\
    0.31&  h=1\\
    -0.155&  h=2 \\
    0&  h \geq 3
  \end{cases}$$
  
## Problem 5 {.unnumbered} 

Assume that the price of an asset at close of trading yesterday was $\$300$ and its volatility was estimated as $1.3\%$ per day. The price at the close of trading today is $\$298$. Update the volatility estimate by using the following methods:

### Question a {.unnumbered} 

EWMA with $\lambda=0.94$.

>$$\begin{align*}
\widehat{\sigma}_t^2 &=\lambda\cdot\widehat{\sigma}_{t-1}^2+(1-\lambda)\cdot y_{t-1}^2 \\
&=0.94\cdot0.013^2+0.06\cdot\left( \ln \frac{298}{300}\right)^2 \\
&\approx1.6153\cdot10^{-4} \\
\rightarrow \widehat{\sigma}_t &\approx0.0127
\end{align*}$$

>Answer: The volatility estimate by using the EWMA with $\lambda=0.94$ is $1.27\%$.

### Question b {.unnumbered} 

The GARCH$(1,1)$ model with $\omega=2\cdot10^{-6},\alpha=0.04,\beta=0.94$.

>$$\begin{align*}
\widehat{\sigma}_t^2 &=\omega+\alpha\cdot y_{t-1}^2+\beta\cdot\widehat{\sigma}_{t-1}^2 \\
&=2\cdot10^{-6}+0.04\cdot\left( \ln \frac{300}{298}\right)^2+0.94\cdot0.013^2 \\
&\approx1.6265\cdot10^{-4} \\
\rightarrow \widehat{\sigma}_t &\approx0.0128
\end{align*}$$
    
>Answer: The volatility estimate by using the GARCH$(1,1)$ model with the given parameters is $1.28\%$.

## Problem 6 {.unnumbered} 

Suppose that the parameters in a GARCH$(1,1)$ model are $\alpha=0.03,\beta=0.95,\omega=2\cdot10^{-6}.$

### Question a {.unnumbered} 

What is the long-run average volatility?

>The long-run average volatility is
>$$\begin{align*}
\sigma &=\sqrt{\frac{\omega}{1-\alpha-\beta}} \\
&=\sqrt{\frac{2\cdot10^{-6}}{1-0.03-0.95}} \\
&=0.01
\end{align*}$$

>Answer: The long-run average volatility is $1\%$.
    
### Question b {.unnumbered} 

If the current volatility is $1.5\%$ per day, what is the estimate of the volatility in $20$, $40$, and $60$ days?

>The estimate of the volatility in $20$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+20}^2) &=\sigma^2+(\alpha+\beta)^{20}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{20}\cdot(0.015^2-0.01^2) \\
&\approx1.8345\cdot10^{-4} \\
\rightarrow \sigma_{n+20}&\approx0.0135 \\
\end{align*}$$

>The estimate of the volatility in $40$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+40}^2) &=\sigma^2+(\alpha+\beta)^{40}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{40}\cdot(0.015^2-0.01^2) \\
&\approx1.5571\cdot10^{-4} \\
\rightarrow \sigma_{n+40}&\approx0.0125 \\
\end{align*}$$

>The estimate of the volatility in $60$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+60}^2) &=\sigma^2+(\alpha+\beta)^{60}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{60}\cdot(0.015^2-0.01^2) \\
&\approx1.3719\cdot10^{-4} \\
\rightarrow \sigma_{n+60}&\approx0.0117 \\
\end{align*}$$

>Answer: If the current volatility is $1.5\%$ per day, the estimates of the volatility in $20$, $40$, and $60$ days are $1.35\%$, $1.25\%$ and $1.17\%$, respectively.
    
### Question c {.unnumbered} 

What volatility should be used to price $20$, $40$, and $60$-day  options?

With the current volatility being $1.5\%$ per day, we have

>$$\begin{align*}
a &=\ln\frac{1}{\alpha+\beta} \\
&=\ln\frac{1}{0.03+0.95} \\
&\approx0.0202
\end{align*}$$

>The volatility should be used to price $20$-day  options:
$$\begin{align*}
\sigma^2(20) &=252\cdot\left(\sigma^2+\frac{1-e^{-20a}}{20a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-20\cdot0.0202}}{20\cdot0.0202}\cdot(0.015^2-0.01^2)\right)\\
&\approx0.0511 \\
\rightarrow\sigma(20) &\approx0.2261
\end{align*}$$

>The volatility should be used to price $40$-day  options:
$$\begin{align*}
\sigma^2(40) &=252\cdot\left(\sigma^2+\frac{1-e^{-40a}}{40a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-40\cdot0.0202}}{40\cdot0.0202}\cdot(0.015^2-0.01^2)\right)\\
&\approx0.0468 \\
\rightarrow\sigma(40) &\approx0.2164
\end{align*}$$

>The volatility should be used to price $60$-day  options:
$$\begin{align*}
\sigma^2(60) &=252\cdot\left(\sigma^2+\frac{1-e^{-60a}}{60a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-60\cdot0.0202}}{60\cdot0.0202}\cdot(0.015^2-0.01^2)\right)\\
&\approx0.0435 \\
\rightarrow\sigma(60) &\approx0.2085
\end{align*}$$

>Answer: The volatility should be used to price $20$, $40$ and $60$-day options are $22.61\%$, $21.64\%$ and $20.85\%$, respectively.

### Question d {.unnumbered} 

Suppose that there is an event that increases the volatility from $1.5%$ per day to $2%$ per day. Estimate the effect on the volatility in $20$, $40$, and $60$ days.

>The estimate of the volatility in $20$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+20}^2) &=\sigma^2+(\alpha+\beta)^{20}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{20}\cdot(0.02^2-0.01^2) \\
&\approx3.0028\cdot10^{-4} \\
\rightarrow \sigma_{n+20}&\approx 0.0173 \\
\end{align*}$$

>The estimate of the volatility in $40$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+40}^2) &=\sigma^2+(\alpha+\beta)^{40}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{40}\cdot(0.02^2-0.01^2) \\
&\approx2.3371\cdot10^{-4} \\
\rightarrow \sigma_{n+40}&\approx 0.0153\\
\end{align*}$$

>The estimate of the volatility in $60$ days is:
$$\begin{align*}
\mathbb{E}(\sigma_{n+60}^2) &=\sigma^2+(\alpha+\beta)^{60}\cdot(\sigma_n^2-\sigma^2) \\
&=0.01^2+(0.03+0.95)^{60}\cdot(0.02^2-0.01^2) \\
&\approx1.8927\cdot10^{-4} \\
\rightarrow \sigma_{n+60}&\approx 0.0138\\
\end{align*}$$

><ul>If the current volatility is $1.5\%$ per day, the estimates of the volatility in $20$, $40$, and $60$ days are $0.0135$, $0.0125$ and $0.0117$, respectively.</ul>
<ul>If the current volatility is $2\%$ per day, the estimates of the volatility in $20$, $40$, and $60$ days are $0.0173$, $0.0153$ and $0.0138$, respectively.</ul>
Therefore, the volatility in $20$, $40$ and $60$ days increases respectively by 
$$\frac{0.0173}{0.0135}-1\approx0.2814 \\
\frac{0.0153}{0.0125}-1\approx0.2240 \\
\frac{0.0138}{0.0117}-1\approx0.1795$$

>Answer: If there is an event that increases the volatility from $1.5%$ per day to $2%$ per day, the volatility in $20$, $40$ and $60$ days increases by $28.14\%$, $22.4\%$ and $17.95\%$, respectively.
    
### Question e {.unnumbered} 

Estimate by how much the event increases the volatilities used to price $20$, $40$, and $60$-day  options.

With the current volatility being $2\%$ per day, we have

>$$\begin{align*}
a &=\ln\frac{1}{\alpha+\beta} \\
&=\ln\frac{1}{0.03+0.95} \\
&\approx0.0202
\end{align*}$$

>The volatility should be used to price $20$-day  options:
$$\begin{align*}
\sigma^2(20) &=252\cdot\left(\sigma^2+\frac{1-e^{-20a}}{20a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-20\cdot0.0202}}{20\cdot0.0202}\cdot(0.02^2-0.01^2)\right)\\
&\approx0.0874 \\
\rightarrow\sigma(20) &\approx0.2956
\end{align*}$$

>The volatility should be used to price $40$-day  options:
$$\begin{align*}
\sigma^2(40) &=252\cdot\left(\sigma^2+\frac{1-e^{-40a}}{40a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-40\cdot0.0202}}{40\cdot0.0202}\cdot(0.02^2-0.01^2)\right)\\
&\approx0.0771 \\
\rightarrow\sigma(40) &\approx0.2776
\end{align*}$$

>The volatility should be used to price $60$-day  options:
$$\begin{align*}
\sigma^2(60) &=252\cdot\left(\sigma^2+\frac{1-e^{-60a}}{60a}\cdot(\sigma_{n}^2-\sigma^2)\right) \\
&= 252\cdot\left(0.01^2+\frac{1-e^{-60\cdot0.0202}}{60\cdot0.0202}\cdot(0.02^2-0.01^2)\right)\\
&\approx0.0690 \\
\rightarrow\sigma(60) &\approx0.2627
\end{align*}$$

><ul>If the current volatility is $1.5\%$ per day, the volatility should be used to price $20$, $40$ and $60$-day options are $0.2261$, $0.2164$ and $0.2085$, respectively.</ul>
<ul>If the current volatility is $2\%$ per day, the volatility should be used to price $20$, $40$ and $60$-day options are $0.2956$, $0.2776$ and $0.2627$, respectively.</ul>
Therefore, if there is an event that increases the volatility from $1.5%$ per day to $2%$ per day, the volatilities used to price increase by
$$\frac{0.2956}{0.2261}-1\approx0.3074 \\
\frac{0.2776}{0.2164}-1\approx0.2828 \\
\frac{0.2627}{0.2085}-1\approx0.2600$$

>Answer: If there is an event that increases the volatility from $1.5%$ per day to $2%$ per day, the volatilities used to price increase by $30.74\%$, $28.28\%$ and $26\%$, respectively.

## Problem 7 {.unnumbered} 

Consider ARCH$(1)$ process, show that
$$\mathbb{E}(\sigma_{t+s}^2|\mathcal{F}_t)=\frac{1-\alpha^s}{1-\alpha}\cdot\omega+\alpha^s\cdot\sigma_t^2,\forall s\geq1$$

>Consider the following ARCH$(1)$ process:
$$\sigma_{t+1}^2=\omega+\alpha \sigma_t^2\epsilon_t^2$$
We will prove the assertion
$$\mathbb{E}(\sigma_{t+s}^2|\mathcal{F}_t)=\frac{1-\alpha^s}{1-\alpha}\cdot\omega+\alpha^s\sigma_t^2,\forall s\geq1$$ 
by induction on $s$. 

>For $s=1$:
$$\begin{align*}
\mathbb{E}(\sigma_{t+1}^2|\mathcal{F}_t) &=\mathbb{E}(\omega+\alpha \sigma_t^2\epsilon_t^2|\mathcal{F}_t) \\
&=\mathbb{E}(\omega+\alpha \sigma_t^2\epsilon_t^2) \\
&=\mathbb{E}(\omega)+\mathbb{E}(\alpha \sigma_t^2\epsilon_t^2) \\
&=\omega+\alpha \sigma_t^2\mathbb{E}(\epsilon_t^2) \\
&=\frac{1-\alpha^1}{1-\alpha}\cdot\omega+\alpha \sigma_t^2
\end{align*}$$
    
>Assume the assertion holds for some $k\in\mathbb{N}$. That is:
$$\begin{align*}
    \mathbb{E}(\sigma_{t+k+1}^2|\mathcal{F}_t) &= \mathbb{E}(\mathbb{E}(\sigma_{t+k+1}^2|\mathcal{F}_{t+1})|\mathcal{F}_t) \\
    &=\mathbb{E}\left(\left.\frac{1-\alpha^k}{1-\alpha}\cdot\omega+\alpha^k\sigma_{t+1}^2\right|\mathcal{F}_t\right)\\
    &= \frac{1-\alpha^k}{1-\alpha}\cdot\omega+\alpha^k\mathbb{E}(\sigma_{t+1}^2|\mathcal{F}_t) \\
    &=\frac{1-\alpha^k}{1-\alpha}\cdot\omega+\alpha^k(\omega+\alpha \sigma_t^2)\\
    &= \left(\frac{1-\alpha^k}{1-\alpha}+\alpha^k\right)\cdot\omega+\alpha^{k+1}\sigma_t^2 \\
    &=\frac{1-\alpha^{k+1}}{1-\alpha}\cdot\omega+\alpha^{k+1}\sigma_t^2
\end{align*}$$

>Therefore, the assertion holds $\forall s\geq1$. 

## Problem 8 {.unnumbered} 

### Python {.unnumbered}

We can download the SP500 data from internet by using Python.


```python
import pandas_datareader as web
price = web.get_data_yahoo("^GSPC",
                           start = "2009-01-01",
                           end ="2021-12-31")
```

### Question a {.unnumbered} 

Denote $y$ the returns of SP500. Plot the returns and calculate statistical characterizations of returns (e.g., min, max, sd, skewness, kurtosis, acf), use Jarque Berra test, Box test.


```python
# Calculate returns y
import numpy as np
y = np.diff(np.log(price['Adj Close']))
```


```python
# Plot the returns
import matplotlib.pyplot as plt
plt.plot(y)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-27-1.png" width="90%" style="display: block; margin: auto;" />


```python
# Statistical characterizations of returns
from scipy.stats import describe
describe(y)
```

```
#> DescribeResult(nobs=3273, minmax=(-0.12765219747281709, 0.08968323251796306), mean=0.0005081885399690335, variance=0.0001314375361112753, skewness=-0.6777972104478536, kurtosis=13.133833345827092)
```


```python
# Autocorrelation function
from statsmodels.tsa.stattools import acf
acf(y)
```

```
#> array([ 1.        , -0.1437426 ,  0.0873594 , -0.03140535, -0.019112  ,
#>         0.00508723, -0.0803378 ,  0.10808321, -0.094185  ,  0.07152584,
#>        -0.00891499, -0.00466423,  0.01929302, -0.04416083,  0.01990267,
#>        -0.07326844,  0.05878776, -0.00504835, -0.02597924, -0.01826176,
#>        -0.01700834,  0.03833622, -0.06437332,  0.03621343, -0.01485914,
#>        -0.03767194, -0.02067457,  0.02913685, -0.01740174,  0.00977897,
#>         0.00331506,  0.00575203, -0.03713924,  0.01212813, -0.01917425,
#>         0.00639421])
```


```python
# Plot ACF of the returns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-30-3.png" width="90%" style="display: block; margin: auto;" />


```python
# Jarque Berra test
import scipy.stats as stats
stats.jarque_bera(y)
```

```
#> Jarque_beraResult(statistic=23774.964889700783, pvalue=0.0)
```


```python
# Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(y)
```

```
#>        lb_stat     lb_pvalue
#> 1    67.688520  1.914751e-16
#> 2    92.697505  7.430063e-21
#> 3    95.930588  1.164970e-20
#> 4    97.128307  4.018129e-20
#> 5    97.213193  2.042500e-19
#> 6   118.389393  3.550392e-23
#> 7   156.729878  1.563740e-30
#> 8   185.852981  6.065963e-36
#> 9   202.653903  9.199582e-39
#> 10  202.914988  3.979104e-38
```
    
### Question b {.unnumbered} 

Use ADF test for the prices and returns to see which series is stationary.


```python
# Plot price
import matplotlib.pyplot as plt
plt.plot(price['Adj Close'])
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-33-5.png" width="90%" style="display: block; margin: auto;" />


```python
# ADF test for prices
from statsmodels.tsa.stattools import adfuller
result = adfuller(price['Adj Close'])
print('p-value: %f' % result[1])
```

```
#> p-value: 0.998453
```


```python
# Plot the returns
import matplotlib.pyplot as plt
plt.plot(y)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-35-7.png" width="90%" style="display: block; margin: auto;" />


```python
# ADF test for returns
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('p-value: %f' % result[1])
```

```
#> p-value: 0.000000
```

### Question c {.unnumbered} 

Using AR$(1)$ to fit the price and returns y and find the coefficients of these fitted AR$(1)$ model. 


```python
# Using AR(1) to fit the price
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
ar1_price = ARIMA(price['Adj Close'], 
                  order = (1, 0, 0)).fit()
```

```
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
#> /Users/cliex159/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
#>   self._init_dates(dates, freq)
```

```python
print(ar1_price.summary())
```

```
#>                                SARIMAX Results                                
#> ==============================================================================
#> Dep. Variable:              Adj Close   No. Observations:                 3274
#> Model:                 ARIMA(1, 0, 0)   Log Likelihood              -15203.588
#> Date:                Thu, 24 Mar 2022   AIC                          30413.175
#> Time:                        16:39:12   BIC                          30431.456
#> Sample:                             0   HQIC                         30419.722
#>                                - 3274                                         
#> Covariance Type:                  opg                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> const       2159.2460   1157.018      1.866      0.062    -108.467    4426.959
#> ar.L1          0.9998      0.000   2301.214      0.000       0.999       1.001
#> sigma2       630.8779      4.496    140.320      0.000     622.066     639.690
#> ===================================================================================
#> Ljung-Box (L1) (Q):                  87.33   Jarque-Bera (JB):             74201.03
#> Prob(Q):                              0.00   Prob(JB):                         0.00
#> Heteroskedasticity (H):               7.34   Skew:                            -1.12
#> Prob(H) (two-sided):                  0.00   Kurtosis:                        26.21
#> ===================================================================================
#> 
#> Warnings:
#> [1] Covariance matrix calculated using the outer product of gradients (complex-step).
```


```python
# Find the coefficients of the fitted AR(1) model
ar1_price.params[0:2]
```

```
#> const    2159.246035
#> ar.L1       0.999802
#> dtype: float64
```


```python
# Using AR(1) to fit the returns
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
ar1_y = ARIMA(y, 
              order = (1, 0, 0)).fit()
print(ar1_y.summary())
```

```
#>                                SARIMAX Results                                
#> ==============================================================================
#> Dep. Variable:                      y   No. Observations:                 3273
#> Model:                 ARIMA(1, 0, 0)   Log Likelihood               10015.913
#> Date:                Thu, 24 Mar 2022   AIC                         -20025.825
#> Time:                        16:39:13   BIC                         -20007.545
#> Sample:                             0   HQIC                        -20019.279
#>                                - 3273                                         
#> Covariance Type:                  opg                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> const          0.0005      0.000      2.773      0.006       0.000       0.001
#> ar.L1         -0.1437      0.007    -19.320      0.000      -0.158      -0.129
#> sigma2         0.0001   1.32e-06     97.279      0.000       0.000       0.000
#> ===================================================================================
#> Ljung-Box (L1) (Q):                   0.31   Jarque-Bera (JB):             18498.33
#> Prob(Q):                              0.58   Prob(JB):                         0.00
#> Heteroskedasticity (H):               0.97   Skew:                            -0.83
#> Prob(H) (two-sided):                  0.57   Kurtosis:                        14.53
#> ===================================================================================
#> 
#> Warnings:
#> [1] Covariance matrix calculated using the outer product of gradients (complex-step).
```


```python
# Find the coefficients of the fitted AR(1) model
ar1_y.params[0:2]
```

```
#> array([ 0.00050302, -0.14374591])
```

Note that If the residuals have a mean other than zero, then the forecasts are biased. Now using this fact you can check the mean of residuals of AR$(1)$ for price and return to see that you can not apply AR$(1)$ directly for price. 


```python
# Check the mean of residuals
ar1_price.resid[1:].mean()
```

```
#> 1.1800462919613017
```


```python
# Check the mean of residuals
ar1_y.resid[1:].mean()
```

```
#> -3.311666982146061e-06
```

Check independence of residuals fitted AR$(1)$, are they independence or not?


```python
# Plot ACF of residuals
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ar1_y.resid)
plt.show()
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-43-9.png" width="90%" style="display: block; margin: auto;" />

ACF are significant in many lags so the residuals of AR$(1)$ of the returns is correlated and thus is not independence.

### Question d {.unnumbered} 

Fit the returns by ARMA$(3,3)$, compare with AR$(1)$ above, can you conclude that
ARMA$(3,3)$ is better than AR$(1)$?


```python
# Fit the returns by ARMA(3,3)
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
arma33_y = ARIMA(y, 
                 order = (3, 0, 3)).fit()
print(arma33_y.summary())
```

```
#>                                SARIMAX Results                                
#> ==============================================================================
#> Dep. Variable:                      y   No. Observations:                 3273
#> Model:                 ARIMA(3, 0, 3)   Log Likelihood               10024.145
#> Date:                Thu, 24 Mar 2022   AIC                         -20032.290
#> Time:                        16:39:19   BIC                         -19983.542
#> Sample:                             0   HQIC                        -20014.833
#>                                - 3273                                         
#> Covariance Type:                  opg                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> const          0.0005      0.000      2.464      0.014       0.000       0.001
#> ar.L1         -0.0633      2.201     -0.029      0.977      -4.377       4.250
#> ar.L2          0.0372      1.404      0.027      0.979      -2.715       2.789
#> ar.L3         -0.0115      0.374     -0.031      0.976      -0.744       0.721
#> ma.L1         -0.0700      2.201     -0.032      0.975      -4.384       4.244
#> ma.L2          0.0399      1.125      0.035      0.972      -2.166       2.246
#> ma.L3         -0.0131      0.349     -0.038      0.970      -0.697       0.670
#> sigma2         0.0001   1.46e-06     87.390      0.000       0.000       0.000
#> ===================================================================================
#> Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             17882.67
#> Prob(Q):                              0.99   Prob(JB):                         0.00
#> Heteroskedasticity (H):               0.95   Skew:                            -0.75
#> Prob(H) (two-sided):                  0.45   Kurtosis:                        14.35
#> ===================================================================================
#> 
#> Warnings:
#> [1] Covariance matrix calculated using the outer product of gradients (complex-step).
```

AIC from ARMA$(3,3)$ is less than AIC from AR$(1)$ so we can conclude that ARMA$(3,3)$ is better than AR$(1)$.

### Question e {.unnumbered} 

Use ARCH$(1)$, GARCH$(1,1)$ and t-GARCH$(1,1)$ to fit the
volatility of the returns and then give your conclusions.


```python
# ARCH(1)
import arch
from arch import arch_model
arch_fit = arch_model(y, mean = 'Zero', 
                         vol = 'ARCH', 
                         q = 1).fit()
```

```
#> Iteration:      1,   Func. Count:      4,   Neg. LLF: -7592.8746576983285
#> Iteration:      2,   Func. Count:     10,   Neg. LLF: -9873.7669996417
#> Iteration:      3,   Func. Count:     15,   Neg. LLF: 4730401.0010247845
#> Iteration:      4,   Func. Count:     21,   Neg. LLF: -10296.047927667008
#> Iteration:      5,   Func. Count:     23,   Neg. LLF: -10296.047927666674
#> Optimization terminated successfully    (Exit mode 0)
#>             Current function value: -10296.047927667008
#>             Iterations: 5
#>             Function evaluations: 23
#>             Gradient evaluations: 5
```

```python
print(arch_fit.summary())
```

```
#>                         Zero Mean - ARCH Model Results                        
#> ==============================================================================
#> Dep. Variable:                      y   R-squared:                       0.000
#> Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
#> Vol Model:                       ARCH   Log-Likelihood:                10296.0
#> Distribution:                  Normal   AIC:                          -20588.1
#> Method:            Maximum Likelihood   BIC:                          -20575.9
#>                                         No. Observations:                 3273
#> Date:                Thu, Mar 24 2022   Df Residuals:                     3273
#> Time:                        16:39:20   Df Model:                            0
#>                               Volatility Model                              
#> ============================================================================
#>                  coef    std err          t      P>|t|      95.0% Conf. Int.
#> ----------------------------------------------------------------------------
#> omega      7.9907e-05  4.727e-06     16.905  4.106e-64 [7.064e-05,8.917e-05]
#> alpha[1]       0.3975  6.722e-02      5.914  3.348e-09     [  0.266,  0.529]
#> ============================================================================
#> 
#> Covariance estimator: robust
```


```python
# GARCH(1,1)
import arch
from arch import arch_model
garch_fit = arch_model(y, 
                       mean = 'Zero', 
                       vol = 'GARCH',
                       p = 1, 
                       q = 1).fit()
```

```
#> Iteration:      1,   Func. Count:      5,   Neg. LLF: 3967.6568400522383
#> Iteration:      2,   Func. Count:     14,   Neg. LLF: -10811.544183900245
#> Optimization terminated successfully    (Exit mode 0)
#>             Current function value: -10811.544186499636
#>             Iterations: 6
#>             Function evaluations: 14
#>             Gradient evaluations: 2
```

```python
print(garch_fit.summary())
```

```
#>                        Zero Mean - GARCH Model Results                        
#> ==============================================================================
#> Dep. Variable:                      y   R-squared:                       0.000
#> Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
#> Vol Model:                      GARCH   Log-Likelihood:                10811.5
#> Distribution:                  Normal   AIC:                          -21617.1
#> Method:            Maximum Likelihood   BIC:                          -21598.8
#>                                         No. Observations:                 3273
#> Date:                Thu, Mar 24 2022   Df Residuals:                     3273
#> Time:                        16:39:21   Df Model:                            0
#>                               Volatility Model                              
#> ============================================================================
#>                  coef    std err          t      P>|t|      95.0% Conf. Int.
#> ----------------------------------------------------------------------------
#> omega      4.0254e-06  2.451e-10  1.642e+04      0.000 [4.025e-06,4.026e-06]
#> alpha[1]       0.2008  4.872e-03     41.225      0.000     [  0.191,  0.210]
#> beta[1]        0.7802  9.059e-03     86.132      0.000     [  0.762,  0.798]
#> ============================================================================
#> 
#> Covariance estimator: robust
```


```python
# t-GARCH(1,1)
import arch
from arch import arch_model
tgarch_fit = arch_model(y, 
                        mean = 'Zero', 
                        vol = 'GARCH',
                        p = 1, 
                        q = 1, 
                        dist = 'StudentsT').fit()
```

```
#> Iteration:      1,   Func. Count:      5,   Neg. LLF: -10902.36462172786
#> Optimization terminated successfully    (Exit mode 0)
#>             Current function value: -10902.36462175196
#>             Iterations: 5
#>             Function evaluations: 5
#>             Gradient evaluations: 1
```

```python
print(tgarch_fit.summary())
```

```
#>                           Zero Mean - GARCH Model Results                           
#> ====================================================================================
#> Dep. Variable:                            y   R-squared:                       0.000
#> Mean Model:                       Zero Mean   Adj. R-squared:                  0.000
#> Vol Model:                            GARCH   Log-Likelihood:                10902.4
#> Distribution:      Standardized Student's t   AIC:                          -21796.7
#> Method:                  Maximum Likelihood   BIC:                          -21772.4
#>                                               No. Observations:                 3273
#> Date:                      Thu, Mar 24 2022   Df Residuals:                     3273
#> Time:                              16:39:22   Df Model:                            0
#>                               Volatility Model                              
#> ============================================================================
#>                  coef    std err          t      P>|t|      95.0% Conf. Int.
#> ----------------------------------------------------------------------------
#> omega      2.6331e-06  6.349e-09    414.698      0.000 [2.621e-06,2.646e-06]
#> alpha[1]       0.2000  1.803e-02     11.094  1.346e-28     [  0.165,  0.235]
#> beta[1]        0.7800  1.543e-02     50.537      0.000     [  0.750,  0.810]
#>                               Distribution                              
#> ========================================================================
#>                  coef    std err          t      P>|t|  95.0% Conf. Int.
#> ------------------------------------------------------------------------
#> nu             6.5511      0.173     37.951      0.000 [  6.213,  6.889]
#> ========================================================================
#> 
#> Covariance estimator: robust
```

AIC from t-GARCH$(1,1)$ is the smallest so we can conclude that t-GARCH$(1,1)$ is the best model.

### R {.unnumbered}

We can download the SP500 data from internet by using R.


```r
library(tseries)
library(zoo)
price = get.hist.quote(instrument = "^gspc",
                       start = "2009-01-01",
                       end = ("2021-12-31"),  
                       quote = "AdjClose")
```

```
#> time series starts 2009-01-02
#> time series ends   2021-12-30
```

### Question a {.unnumbered} 

Denote $y$ the returns of SP500. Plot the returns and calculate statistical characterizations of returns (e.g., min, max, sd, skewness, kurtosis, acf), use Jarque Berra test, Box test.


```r
# Calculate returns y
library(tidyverse)
y = price %>% 
  log %>% 
  diff
```


```r
# Plot the returns
y %>% 
  plot
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-50-1.png" width="90%" style="display: block; margin: auto;" />


```r
# Statistical characterizations of returns
library(psych)
y %>% 
  describe %>% 
  select(min,
         max,
         sd,
         skew,
         kurtosis)
```

```
#>      min  max   sd  skew kurtosis
#> X1 -0.13 0.09 0.01 -0.68    13.17
```


```r
# Plot ACF of the returns
y %>% 
  coredata %>% 
  acf
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-52-1.png" width="90%" style="display: block; margin: auto;" />


```r
# Jarque Berra test
library(moments)
y %>% 
  coredata %>% 
  c %>% 
  jarque.test
```

```
#> 
#> 	Jarque-Bera Normality Test
#> 
#> data:  .
#> JB = 23922, p-value < 2.2e-16
#> alternative hypothesis: greater
```


```r
# Box test
library(stats)
y %>% 
  Box.test(lag = 10, 
           type = "Ljung-Box")
```

```
#> 
#> 	Box-Ljung test
#> 
#> data:  .
#> X-squared = 180.1, df = 10, p-value < 2.2e-16
```
    
### Question b {.unnumbered} 

Use ADF test for the prices and returns to see which series is stationary.


```r
# Plot price
price %>% 
  plot
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-55-1.png" width="90%" style="display: block; margin: auto;" />


```r
# ADF test for prices
library(tseries)
price %>% 
  coredata %>%
  adf.test
```

```
#> Augmented Dickey-Fuller Test 
#> alternative: stationary 
#>  
#> Type 1: no drift no trend 
#>       lag  ADF p.value
#>  [1,]   0 2.96    0.99
#>  [2,]   1 3.59    0.99
#>  [3,]   2 3.25    0.99
#>  [4,]   3 3.19    0.99
#>  [5,]   4 3.46    0.99
#>  [6,]   5 3.42    0.99
#>  [7,]   6 3.77    0.99
#>  [8,]   7 3.28    0.99
#>  [9,]   8 3.62    0.99
#> Type 2: with drift no trend 
#>       lag  ADF p.value
#>  [1,]   0 1.25    0.99
#>  [2,]   1 1.73    0.99
#>  [3,]   2 1.48    0.99
#>  [4,]   3 1.41    0.99
#>  [5,]   4 1.62    0.99
#>  [6,]   5 1.57    0.99
#>  [7,]   6 1.80    0.99
#>  [8,]   7 1.46    0.99
#>  [9,]   8 1.67    0.99
#> Type 3: with drift and trend 
#>       lag    ADF p.value
#>  [1,]   0 -1.238   0.900
#>  [2,]   1 -0.564   0.979
#>  [3,]   2 -0.906   0.952
#>  [4,]   3 -0.951   0.947
#>  [5,]   4 -0.667   0.974
#>  [6,]   5 -0.691   0.971
#>  [7,]   6 -0.351   0.989
#>  [8,]   7 -0.794   0.962
#>  [9,]   8 -0.452   0.984
#> ---- 
#> Note: in fact, p.value = 0.01 means p.value <= 0.01
```


```r
# Plot the returns
y %>% 
  plot
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-57-1.png" width="90%" style="display: block; margin: auto;" />


```r
# ADF test for returns
library(tseries)
y %>% 
  coredata %>%
  adf.test
```

```
#> Augmented Dickey-Fuller Test 
#> alternative: stationary 
#>  
#> Type 1: no drift no trend 
#>       lag   ADF p.value
#>  [1,]   0 -65.9    0.01
#>  [2,]   1 -40.3    0.01
#>  [3,]   2 -33.1    0.01
#>  [4,]   3 -29.5    0.01
#>  [5,]   4 -26.1    0.01
#>  [6,]   5 -25.6    0.01
#>  [7,]   6 -21.3    0.01
#>  [8,]   7 -21.2    0.01
#>  [9,]   8 -19.2    0.01
#> Type 2: with drift no trend 
#>       lag   ADF p.value
#>  [1,]   0 -66.1    0.01
#>  [2,]   1 -40.4    0.01
#>  [3,]   2 -33.3    0.01
#>  [4,]   3 -29.6    0.01
#>  [5,]   4 -26.3    0.01
#>  [6,]   5 -25.8    0.01
#>  [7,]   6 -21.5    0.01
#>  [8,]   7 -21.5    0.01
#>  [9,]   8 -19.4    0.01
#> Type 3: with drift and trend 
#>       lag   ADF p.value
#>  [1,]   0 -66.1    0.01
#>  [2,]   1 -40.4    0.01
#>  [3,]   2 -33.3    0.01
#>  [4,]   3 -29.6    0.01
#>  [5,]   4 -26.3    0.01
#>  [6,]   5 -25.8    0.01
#>  [7,]   6 -21.5    0.01
#>  [8,]   7 -21.5    0.01
#>  [9,]   8 -19.4    0.01
#> ---- 
#> Note: in fact, p.value = 0.01 means p.value <= 0.01
```

### Question c {.unnumbered} 

Using AR$(1)$ to fit the price and returns y and find the coefficients of these fitted AR$(1)$ model. 


```r
# Using AR(1) to fit the price
library(stats)
ar1_price = price %>%
  arima(c(1,0,0))
```


```r
# Find the coefficients of the fitted AR(1) model
ar1_price %>%
  coefficients
```

```
#>          ar1    intercept 
#>    0.9999917 2159.3788250
```


```r
# Using AR(1) to fit the returns
library(stats)
ar1_y = y %>% 
  arima(c(1,0,0))
```


```r
# Find the coefficients of the fitted AR(1) model
ar1_y %>% 
  coefficients
```

```
#>           ar1     intercept 
#> -0.1576312868  0.0005086707
```

Note that If the residuals have a mean other than zero, then the forecasts are biased. Now using this fact you can check the mean of residuals of AR$(1)$ for price and return to see that you can not apply AR$(1)$ directly for price. 


```r
# Check the mean of residuals
ar1_price %>% 
  residuals %>% 
  coredata %>% 
  na.omit %>% 
  mean
```

```
#> [1] 1.131751
```


```r
# Check the mean of residuals
ar1_y %>% 
  residuals %>% 
  coredata %>% 
  na.omit %>% 
  mean
```

```
#> [1] -3.824746e-06
```

Check independence of residuals fitted AR$(1)$, are they independence or not?


```r
# Plot ACF of the residuals
library(stats)
ar1_y %>% 
  residuals %>% 
  coredata %>% 
  na.omit %>% 
  acf
```

<img src="02-cross-refs_files/figure-html/unnamed-chunk-65-1.png" width="90%" style="display: block; margin: auto;" />

ACF are significant in many lags so the residuals of AR$(1)$ of the returns is correlated and thus is not independence.

### Question d {.unnumbered} 

Fit the returns by ARMA$(3,3)$, compare with AR$(1)$ above, can you conclude that
ARMA$(3,3)$ is better than AR$(1)$?


```r
# Fit the returns by ARMA(3,3)
library(stats)
arma33_y = y %>%
  arima(c(3,0,3))
arma33_y$aic < ar1_y$aic
```

```
#> [1] TRUE
```

AIC from ARMA$(3,3)$ is less than AIC from AR$(1)$ so we can conclude that ARMA$(3,3)$ is better than AR$(1)$.

### Question e {.unnumbered} 

Use ARCH$(1)$, GARCH$(1,1)$ and t-GARCH$(1,1)$ (using package rugarch) to fit the
volatility of the returns and then give your conclusions.


```r
# ARCH(1)
library(fGarch)
arch.fit = garchFit(~garch(1,0), data = y, trace = F)
summary(arch.fit)
```

```
#> 
#> Title:
#>  GARCH Modelling 
#> 
#> Call:
#>  garchFit(formula = ~garch(1, 0), data = y, trace = F) 
#> 
#> Mean and Variance Equation:
#>  data ~ garch(1, 0)
#> <environment: 0x7fc59153aae8>
#>  [data = y]
#> 
#> Conditional Distribution:
#>  norm 
#> 
#> Coefficient(s):
#>         mu       omega      alpha1  
#> 8.6504e-04  7.8455e-05  4.1419e-01  
#> 
#> Std. Errors:
#>  based on Hessian 
#> 
#> Error Analysis:
#>         Estimate  Std. Error  t value Pr(>|t|)    
#> mu     8.650e-04   1.640e-04    5.276 1.32e-07 ***
#> omega  7.845e-05   2.664e-06   29.452  < 2e-16 ***
#> alpha1 4.142e-01   3.821e-02   10.839  < 2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Log Likelihood:
#>  10305.09    normalized:  3.15044 
#> 
#> Description:
#>  Thu Mar 24 16:39:26 2022 by user:  
#> 
#> 
#> Standardised Residuals Tests:
#>                                 Statistic p-Value    
#>  Jarque-Bera Test   R    Chi^2  4427.257  0          
#>  Shapiro-Wilk Test  R    W      0.9298395 0          
#>  Ljung-Box Test     R    Q(10)  22.19605  0.01413642 
#>  Ljung-Box Test     R    Q(15)  31.72511  0.007016061
#>  Ljung-Box Test     R    Q(20)  39.5145   0.005750492
#>  Ljung-Box Test     R^2  Q(10)  663.3026  0          
#>  Ljung-Box Test     R^2  Q(15)  880.8894  0          
#>  Ljung-Box Test     R^2  Q(20)  1068.422  0          
#>  LM Arch Test       R    TR^2   427.7749  0          
#> 
#> Information Criterion Statistics:
#>       AIC       BIC       SIC      HQIC 
#> -6.299046 -6.293458 -6.299048 -6.297045
```


```r
# GARCH(1,1)
library(rugarch)
garch_spec = ugarchspec(variance.model = list(model = "sGARCH",
                                           garchOrder = c(1,1)),
                       mean.model = list(armaOrder = c(0,0),
                                       include.mean = T,
                                       distribution.model = "norm"))
garch_fit = ugarchfit(garch_spec, y %>% coredata %>% c)
garch_fit
```

```
#> 
#> *---------------------------------*
#> *          GARCH Model Fit        *
#> *---------------------------------*
#> 
#> Conditional Variance Dynamics 	
#> -----------------------------------
#> GARCH Model	: sGARCH(1,1)
#> Mean Model	: ARFIMA(0,0,0)
#> Distribution	: norm 
#> 
#> Optimal Parameters
#> ------------------------------------
#>         Estimate  Std. Error  t value Pr(>|t|)
#> mu      0.000805    0.000129   6.2607 0.000000
#> omega   0.000003    0.000001   2.6538 0.007958
#> alpha1  0.170394    0.014539  11.7194 0.000000
#> beta1   0.802782    0.015565  51.5761 0.000000
#> 
#> Robust Standard Errors:
#>         Estimate  Std. Error  t value Pr(>|t|)
#> mu      0.000805    0.000185   4.3598 0.000013
#> omega   0.000003    0.000008   0.4560 0.648389
#> alpha1  0.170394    0.025679   6.6356 0.000000
#> beta1   0.802782    0.058438  13.7374 0.000000
#> 
#> LogLikelihood : 10826.74 
#> 
#> Information Criteria
#> ------------------------------------
#>                     
#> Akaike       -6.6174
#> Bayes        -6.6099
#> Shibata      -6.6174
#> Hannan-Quinn -6.6147
#> 
#> Weighted Ljung-Box Test on Standardized Residuals
#> ------------------------------------
#>                         statistic p-value
#> Lag[1]                      4.129 0.04216
#> Lag[2*(p+q)+(p+q)-1][2]     4.271 0.06377
#> Lag[4*(p+q)+(p+q)-1][5]     5.361 0.12664
#> d.o.f=0
#> H0 : No serial correlation
#> 
#> Weighted Ljung-Box Test on Standardized Squared Residuals
#> ------------------------------------
#>                         statistic p-value
#> Lag[1]                     0.3257  0.5682
#> Lag[2*(p+q)+(p+q)-1][5]    2.0941  0.5964
#> Lag[4*(p+q)+(p+q)-1][9]    2.8388  0.7853
#> d.o.f=2
#> 
#> Weighted ARCH LM Tests
#> ------------------------------------
#>             Statistic Shape Scale P-Value
#> ARCH Lag[3]  0.009343 0.500 2.000  0.9230
#> ARCH Lag[5]  0.933101 1.440 1.667  0.7529
#> ARCH Lag[7]  1.242758 2.315 1.543  0.8712
#> 
#> Nyblom stability test
#> ------------------------------------
#> Joint Statistic:  3.0795
#> Individual Statistics:              
#> mu     0.06252
#> omega  0.06964
#> alpha1 0.34031
#> beta1  0.80804
#> 
#> Asymptotic Critical Values (10% 5% 1%)
#> Joint Statistic:     	 1.07 1.24 1.6
#> Individual Statistic:	 0.35 0.47 0.75
#> 
#> Sign Bias Test
#> ------------------------------------
#>                    t-value      prob sig
#> Sign Bias           3.0094 2.637e-03 ***
#> Negative Sign Bias  0.6287 5.296e-01    
#> Positive Sign Bias  1.6317 1.028e-01    
#> Joint Effect       24.5022 1.962e-05 ***
#> 
#> 
#> Adjusted Pearson Goodness-of-Fit Test:
#> ------------------------------------
#>   group statistic p-value(g-1)
#> 1    20     151.8    9.633e-23
#> 2    30     174.9    8.735e-23
#> 3    40     195.7    9.306e-23
#> 4    50     222.6    5.796e-24
#> 
#> 
#> Elapsed time : 0.21032
```


```r
# TGARCH(1,1)
library(rugarch)
tgarch_spec = ugarchspec(variance.model = list(model = "sGARCH",
                                           garchOrder = c(1,1)),
                       mean.model = list(armaOrder = c(0,0),
                                       include.mean = T,
                                       distribution.model = "std"))
tgarch_fit = ugarchfit(tgarch_spec, y %>% coredata %>% c)
tgarch_fit
```

```
#> 
#> *---------------------------------*
#> *          GARCH Model Fit        *
#> *---------------------------------*
#> 
#> Conditional Variance Dynamics 	
#> -----------------------------------
#> GARCH Model	: sGARCH(1,1)
#> Mean Model	: ARFIMA(0,0,0)
#> Distribution	: norm 
#> 
#> Optimal Parameters
#> ------------------------------------
#>         Estimate  Std. Error  t value Pr(>|t|)
#> mu      0.000805    0.000129   6.2607 0.000000
#> omega   0.000003    0.000001   2.6538 0.007958
#> alpha1  0.170394    0.014539  11.7194 0.000000
#> beta1   0.802782    0.015565  51.5761 0.000000
#> 
#> Robust Standard Errors:
#>         Estimate  Std. Error  t value Pr(>|t|)
#> mu      0.000805    0.000185   4.3598 0.000013
#> omega   0.000003    0.000008   0.4560 0.648389
#> alpha1  0.170394    0.025679   6.6356 0.000000
#> beta1   0.802782    0.058438  13.7374 0.000000
#> 
#> LogLikelihood : 10826.74 
#> 
#> Information Criteria
#> ------------------------------------
#>                     
#> Akaike       -6.6174
#> Bayes        -6.6099
#> Shibata      -6.6174
#> Hannan-Quinn -6.6147
#> 
#> Weighted Ljung-Box Test on Standardized Residuals
#> ------------------------------------
#>                         statistic p-value
#> Lag[1]                      4.129 0.04216
#> Lag[2*(p+q)+(p+q)-1][2]     4.271 0.06377
#> Lag[4*(p+q)+(p+q)-1][5]     5.361 0.12664
#> d.o.f=0
#> H0 : No serial correlation
#> 
#> Weighted Ljung-Box Test on Standardized Squared Residuals
#> ------------------------------------
#>                         statistic p-value
#> Lag[1]                     0.3257  0.5682
#> Lag[2*(p+q)+(p+q)-1][5]    2.0941  0.5964
#> Lag[4*(p+q)+(p+q)-1][9]    2.8388  0.7853
#> d.o.f=2
#> 
#> Weighted ARCH LM Tests
#> ------------------------------------
#>             Statistic Shape Scale P-Value
#> ARCH Lag[3]  0.009343 0.500 2.000  0.9230
#> ARCH Lag[5]  0.933101 1.440 1.667  0.7529
#> ARCH Lag[7]  1.242758 2.315 1.543  0.8712
#> 
#> Nyblom stability test
#> ------------------------------------
#> Joint Statistic:  3.0795
#> Individual Statistics:              
#> mu     0.06252
#> omega  0.06964
#> alpha1 0.34031
#> beta1  0.80804
#> 
#> Asymptotic Critical Values (10% 5% 1%)
#> Joint Statistic:     	 1.07 1.24 1.6
#> Individual Statistic:	 0.35 0.47 0.75
#> 
#> Sign Bias Test
#> ------------------------------------
#>                    t-value      prob sig
#> Sign Bias           3.0094 2.637e-03 ***
#> Negative Sign Bias  0.6287 5.296e-01    
#> Positive Sign Bias  1.6317 1.028e-01    
#> Joint Effect       24.5022 1.962e-05 ***
#> 
#> 
#> Adjusted Pearson Goodness-of-Fit Test:
#> ------------------------------------
#>   group statistic p-value(g-1)
#> 1    20     151.8    9.633e-23
#> 2    30     174.9    8.735e-23
#> 3    40     195.7    9.306e-23
#> 4    50     222.6    5.796e-24
#> 
#> 
#> Elapsed time : 0.360929
```

AIC from t-GARCH$(1,1)$ is the smallest so we can conclude that t-GARCH$(1,1)$ is the best model.
