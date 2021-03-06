# (PART) Midterm Test {.unnumbered}

# Sem 2 2020-2021

## Problem 1 {.unnumbered}

The prices and dividends of a stock are given as follows.


| time | $P_t$ | $D_t$ |
|------|:-----:|------:|
| 1 | 82 | 0.1 |
| 2 | 85 | 0.1 |
| 3 | 83 | 0.1 |
| 4 | 87 | 0.125 |

### Question a {.unnumbered}

Determine $R_2$ and $R_4(3)$.

>
$$\begin{align*}
R_2&=\frac{P_2-P_1+d_2}{P_1} \\
&=\frac{85-82+0.1}{82} \\
&=0.038 \\
R_3&=\frac{P_3-P_2+d_3}{P_2} \\
&=\frac{83-85+0.1}{85} \\
&=-0.022 \\
R_4&=\frac{P_4-P_3+d_4}{P_3} \\
&=\frac{87-83+0.125}{83} \\
&=0.050 \\
\end{align*}$$

>
$$\begin{align*}
R_4(3)&=(1+R_4)(1+R_3)(1+R_2)-1 \\
R_4(3)&=(1+0.038)(1-0.022)(1+0.050)-1 \\
&\approx 0.066
\end{align*}$$

>Answer: $R_2 \approx 0.042$ and $R_4(3) \approx 0.148$.

### Question b {.unnumbered}

Determine $r_3$.

>$$\begin{align*}
r_3&=\ln(1+R_3) \\
&\approx R_3 \\
&\approx-0.022
\end{align*}$$

>Answer: $r_3 \approx -0.022$

## Problem 2 {.unnumbered}

The daily log returns on a stock are normally distributed with mean $0.001$ and standard deviation $0.02$. The stock price now is $\$99$. What is the probability that it will exceed $\$103$ after $10$ trading days?

>
$$\begin{align*}
\ln \left( \frac{P_{10}}{P_0} \right) &= r_{10}(10) \\
\ln  P_{10} - \ln P_0 &= \sum_{t=1}^{10}r_t \\
&\sim 10 \cdot \mathcal{N}(0.001, 0.02^2) \\
&\sim  \mathcal{N}(10 \cdot 0.001, 10 \cdot 0.02^2) \\
\end{align*}$$

>
$$\begin{align*}
\rightarrow \ln(P_{10})&=\ln(P_{0})+r_{10}(10) \\
&\sim \ln(99)+\mathcal{N}(10 \cdot 0.001, 10 \cdot 0.02^2) \\
&\sim \mathcal{N}(\ln(99)+10 \cdot 0.001, 10 \cdot 0.02^2) \\
\end{align*}$$

>
$$\begin{align*}
\mathcal{P}(P_{10}>103)&=\mathcal{P}(\ln(P_{10})>\ln(103)) \\
&=\mathcal{P} \left(\frac{\ln(P_{10})-(\ln(99)+10 \cdot 0.001)}{\sqrt{10 \cdot 0.02^2}} > \frac{\ln(103)-(\ln(99)+10 \cdot 0.001)}{\sqrt{10 \cdot 0.02^2}} \right) \\
&=\mathcal{P} \left(\mathcal{Z} > \frac{\ln(103)-(\ln(99)+10 \cdot 0.001)}{\sqrt{10 \cdot 0.02^2}} \right) \\
&=\mathcal{P} \left(\mathcal{Z} < -\frac{\ln(103)-(\ln(99)+10 \cdot 0.001)}{\sqrt{10 \cdot 0.02^2}} \right) \\
&=0.31983
\end{align*}$$

>Answer: The probability that its price exceeds $\$103$ after 10 trading days is 31.983\%.

## Problem 3 {.unnumbered}

a. Write down the definition of a weakly stationary process.

>A process is weakly stationary if its mean, variance, and covariance are unchanged by time shifts. More precisely, $X_1, X_2, ...$ is a weakly stationary process if

1. $\mathbb{E}(X_t)=\mu, \forall t$
2. $Var(X_t) = \sigma_2$ (a positive finite constant) for all $t$.
3. $Cov(X_t, X_s) = \gamma(|t ??? s|), \forall t, s$ and some function $\gamma$.

b. Write down the first-order Autoregressive model ($AR(1)$) of the time series $X_t$. Write down the formulas of mean $\mathbb{E}(X_t)$, variance $Var(X_t)$ and the correlation function $\rho(h)$ between observations $h$ time periods, and find the condition such that $X_t$ is a weakly stationary.

>The time series $X = (X_t)$ is called AR(1) if the value of X at time t is a linear function of the value of $X$ at time $t ??? 1$ as follows
$$X_t=\delta+\phi_1 X_{t-1}+w_t$$

1. The errors $w_t \sim \mathcal{N}(0,\sigma_w^2)$ are i.i.d.
2. $w_t$ is independent of $X_t$.
3. $\phi_1<1$. This condition guarantees that $X_t$ is weakly stationary.

>$$\begin{align*} 
&\mu=\mathbb{E}(X_t)=\frac{\delta}{1-\phi_1} \\
\\
&Var(X_t)=\frac{\sigma_w^2}{1-\phi_1^2} \\
\\
&Cov(X_t,X_{t+h})=\gamma(h)=\phi_1^h \cdot \frac{\sigma_w^2}{1-\phi_1^2} \\
\\
&\rho(h)=\phi_1^h
\end{align*}$$

## Problem 4 {.unnumbered}

Assume that the price of an asset at close of trading yesterday was $\$110$ and its volatility was estimated as $1.1\%$ per day. The price at the close of trading today is $\$108.5$. Update the volatility estimate by using the following methods:

### Question a {.unnumbered} 

EWMA with $\lambda=0.9$.

>$$\begin{align*}
\widehat{\sigma}_t^2 &=\lambda\cdot\widehat{\sigma}_{t-1}^2+(1-\lambda)\cdot y_{t-1}^2 \\
&=0.9\cdot0.011^2+(1-0.9)\cdot\left( \ln \frac{108.5}{110}\right)^2 \\
\rightarrow \widehat{\sigma}_t &\approx0.0113
\end{align*}$$

>Answer: The volatility estimate by using the EWMA with $\lambda=0.9$ is $1.13\%$.

### Question b {.unnumbered} 

The GARCH$(1,1)$ model with $\omega=1\cdot10^{-6},\alpha=0.05,\beta=0.94$.

>$$\begin{align*}
\widehat{\sigma}_t^2 &=\omega+\alpha\cdot y_{t-1}^2+\beta\cdot\widehat{\sigma}_{t-1}^2 \\
&=1\cdot10^{-6}+0.05\cdot\left( \ln \frac{108.5}{110}\right)^2+0.94\cdot0.011^2 \\
\rightarrow \widehat{\sigma}_t &\approx0.0111
\end{align*}$$
    
>Answer: The volatility estimate by using the GARCH$(1,1)$ model with the given parameters is $1.11\%$.

## Problem 5 {.unnumbered} 

The loss $L$ of an investment has probability distribution


|$L$ | $-200$ | $-150$ | $0$ | $50$ | $100$ | $170$ |
|-----|--------|--------|-----|------|-------|-------|
| $p$ | $0.1681$ | $0.3602$ | $0.3087$ | $0.1323$ | $0.0284$ | $0.0023$ |

Compute Value-at-Risk at confidence level $\alpha = 99 \%$

>The cumulative distribution function of the loss $L$ is
$$\begin{align*}
F(x) &=\begin{cases}
      0 & x \leq -200 \\
      0.1681 & -200 \leq x \leq -150 \\
      0.1681+0.3602 & -150 \leq x \leq 0 \\
      0.1681+0.3602+0.3087 & 0 \leq x \leq 50 \\
      0.1681+0.3602+0.3087+0.1323 & 50 \leq x \leq 100 \\
      0.1681+0.3602+0.3087+0.1323+0.0284 & 100 \leq x \leq170 \\
      0.1681+0.3602+0.3087+0.1323+0.0284+0.0023 & 170 \leq x \\
    \end{cases} \\
    &=\begin{cases}
      0 & x \leq -200 \\
      0.1681 & -200 \leq x \leq -150 \\
      0.5283 & -150 \leq x \leq 0 \\
      0.837 & 0 \leq x \leq 50 \\
      0.9693 & 50 \leq x \leq 100 \\
      0.9977 & 100 \leq x \leq170 \\
      1 & 170 \leq x \\
    \end{cases}
\end{align*}$$

>$$VaR_{0.99}(L_{100}) = \inf \{ x:F_L(x) \geq 0.99 \} = 100 $$

>Answer: $VaR_{0.99}(L_{100})=100$

## Problem 6 {.unnumbered} 
 
 You hold a portfolio consisting of a long position of 1 share of stock $S$. The stock price today is $P_0=\$100$. Let $X_t$ denote the log return of day $t$. The daily log returns are assumed to be independent and normally distributed with zero mean and standard deviation $\sigma = 0.1$. Let $L_{100}$ denote the loss from today until 100 trading days before deciding what to do with the portfolio.
 
 a. Prove that $L_{100} = -P_0(e^{\sum_{i=1}^{100}}-1).$

>$$\begin{align*}
-P_0(e^{\sum_{i=1}^{100} X_i}-1) &= P_0 - P_0e^{X_1+X_2+...+X_{100}} \\
&= P_0 - P_0e^{\sum_{i=1}^{100} \ln \frac{P_1}{P_0} + \ln \frac{P_2}{P_1}+...+\ln \frac{P_{100}}{P_{99}}} \\
&= P_0 - P_0e^{ \ln \frac{P_{100}}{P_0}} \\
&= P_0 - P_0 \cdot \frac{P_{100}}{P_0} \\
&= P_0 - P_{100} \\
&= L_{100}
\end{align*}$$

b. From question a) compute $VaR_{0.99}(L_{100})$

>
$$\begin{align*}
L_{100} &= -P_0(e^{\sum_{i=1}^{100} X_i}-1) \\ 
\rightarrow \ln \left( 1 - \frac{L_{100}}{P_0} \right) &= \sum_{i=1}^{100} X_i \\
& \sim 100 \cdot \mathcal{N}(0,0.1^2) \\
& \sim  \mathcal{N}(100 \cdot 0,100 \cdot 0.1^2) \\
\rightarrow 1 - \frac{L_{100}}{P_0} &\sim \log \mathcal{N}(100 \cdot 0,100 \cdot 0.1^2) \\
\rightarrow L_{100} &\sim P_0 - P_0 \cdot \log \mathcal{N}(100 \cdot 0,100 \cdot 0.1^2) \\
\rightarrow F_{L_{100}}(x) &= 100 - 100 \cdot \Phi \left(\frac{\ln x - 100 \cdot 0}{100 \cdot 0.1^2} \right)
\end{align*}$$

>$$\begin{align*}
F_{L_{100}}(x) &\geq 0.99 \\
\rightarrow 100 - 100 \cdot \Phi \left(\frac{\ln x - 100 \cdot 0}{100 \cdot 0.1^2} \right) &\geq 0.99 \\
\rightarrow \Phi \left(\frac{\ln x - 100 \cdot 0}{100 \cdot 0.1^2} \right) &\leq \frac{100-0.99}{100} \\
\rightarrow \frac{\ln x - 100 \cdot 0}{100 \cdot 0.1^2} &\leq \Phi^{-1}(0.99) \\
\rightarrow x &\leq e^{100 \cdot 0 + 100 \cdot 0.1^2 \cdot \Phi^{-1}(0.99)} \\
\end{align*}$$

>$$\begin{align*}
VaR_{0.99}(L_{100}) &= \inf \{x:F_{L_{100}}(x) \geq 0.99  \} \\
&=e^{2.33} 
\end{align*}$$

>Answer: $VaR_{0.99}(L_{100})=e^{2.33}$
