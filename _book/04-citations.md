# (PART) Chapter 3 Extreme Value Theory {.unnumbered}

# Type of Tails

* In FRM we focus principally on events that occur with a 1% or 5%
probability. This is fine for most day-to-day applications in financial institutions, and methods such as GARCH and historical
simulation are well suited to provide VaR for these purposes.
* In most risk analysis we are concerned with the negative observations in the lower tails, hence to follow the convention, we can pre-multiply returns by -1
* In most risk applications, we do not need to focus on the entire distribution. Since all we care about are large losses, which usually belong in the tails. For example, GARCH modeling is done with the entire distribution of returns.
* EVT, on the other hand, focuses explicitly on analyzing the tail regions of distributions (i.e., the probability of uncommon events).
* Furthermore, there is no reason to believe the distribution of returns is symmetric; the upper and lower tails do not have the same thickness or shape. In most cases, the upper tail of returns is thinner than lower tail. For example, the upper tail of return of Microsoft is thinner than the lower tail.
* EVT can be useful in such situations as it enables us to explicitly
identify the type of asymmetry in the extreme tails.
* In most risk applications we do not need to focus on the entire
distribution of returns since all we care about are large losses,
which usually belong in the tails.
* The main result of EVT states that, regardless of the overall shape
of the distribution,the tails of all distributions fall into one of three
categories as long as the distribution of an asset return series
does not change over time. This means that for risk applications
we only need to focus on one of these three categories:
- Weibull Thin tails where the distribution has a finite endpoint (e.g.,
the distribution of mortality and insurance/re-insurance claims).
- Gumbel Tails decline exponentially (e.g., the normal and log-normal
distributions).
- Frechet Tails decline by a power law; such tails are known as “fat
tails” (e.g., the Student-t and Pareto distributions).

The following figure shows the distributions of Weibull, Gumbel and
Frechet distributions

* From the figure above we see that: (i) the Weibull clearly has a finite endpoint. (ii) the Frechet tail is thicker than the Gumbel’s.
* In most applications in finance, we know that returns are fat tailed

# GEV distribution

Quantitative financial risk management is concerned with maximal
losses (worst-case losses). Let Xi be i.i.d with distribution F
continuous. Then the block maximum is given by

$$M_n := \max(X_1,X_2,...,X_n)$$

## Maximum domain of attraction {.unnumbered} 

Suppose we find normalizing sequences of real numbers $c_n > 0$ and
$d_n$, such that $\frac{M_n − d_n}{c_n}$ converge in distribution, i.e.,

$$\begin{align*}
P\left(\frac{M_n-d_n}{c_n} \leq x \right) &= P(M_n \leq c_nx+d_n) \\
&= P(M_i \leq c_nx+d_n,i=1,2,...,n) \\
&= F^n(c_nx+d_n) \rightarrow H(x), n \rightarrow \infty
\end{align*}$$
for some non-degenerate df $H$ (not a unit jump). Then F is in the
maximum domain of attraction of $H(F \in MDA(H))$.

## The (standard) generalized extreme value (GEV) {.unnumbered} 

The (standard) generalized extreme value (GEV)
distribution is given by

$$H_\xi(x)=\begin{equation}
    \begin{cases}
      e^{-(1+ \xi x)^{\frac{1}{\xi}}}, & \xi \neq 0 \ \\
      e^{e^{-x}}, & \xi = 0
    \end{cases}\,
\end{equation}$$

where $1 + \xi x > 0$. Depending on the value of $\xi, H_{\xi}$ become one of the
three distributions:

* if $\xi > 0$, $H_{\xi}$ is the Frechet
* if $\xi < 0$, $H_{\xi}$ is the Weibull
* if $\xi = 0$, $H_{\xi}$ is the Gumbel

Fisher-Tippet and Gnedenko theorems

* The theorems state that the maximum of a sample of properly normalized IID random variables converges in distribution to one of the three possible distributions: the Weibull, Gumbel or the Frechet
* Let $X_1, X_2, ..., X_T$ denote IID random variables (RVs) and the term
MT indicate maxima in sample of size $T$
The standardized distribution of maxima, $M_T := \max(X_1, X_2, ..., X_T)$, is

$$\lim_{T \to \infty} P \left( \frac{M_T-a_T}{b_T} \leq x \right) = H(x)$$

where the constants $a_T$ and $b_T > 0$ exist and are defined as $a_T = T \mathbb{E}(X_1)$ and $b_T = \sqrt{Var(X_1)}$

Example 1(Exponential distribution)
Let $X_i \sim Exp(\lambda)$, choosing $c_n = \frac{1}{\lambda}$ and $d_n = \frac{\log(n)}{\lambda}$, we obtain

$$F^n(c_nx + d_n) = \left(1 − \frac{e^{−x}}{n} \right)^n \\
\rightarrow e^(−e^{−x}) = H(x) (Gumbell)$$

Example 2. (Pareto distribution)
Let $X_i$ be i.i.d and have Pareto distribution with

$$F(x) = 1 − \left( \frac{\kappa}{\kappa +x} \right)^{\theta}, x \geq 0, \theta, \kappa > 0$$

Chosing $c_n = \frac{\kappa n^{\frac{1}{\theta}}}{\theta}$ and $d_n = \kappa(n^{1θ }− 1)$. Then we get
$$F^n(c_nx+d_n)=\left( 1+\frac{-\left(1+\frac{x}{\theta} \right)^{-\theta}}{n} \right)^n \\
\rightarrow e^{-\left( 1+\frac{x}{\theta} \right)-\theta}=H_{\frac{1}{\theta}}(x)$$

# Asset returns and Fat tails

* The term “fat tails” can have several meanings, the most common being “extreme outcomes occur more frequently than predicted by normal distribution”.
* The most frequent definition one may encounter is Kurtosis, but it is not always accurate at indicating the presence of fat tails $(Kur > 3)$.
* This is because kurtosis is more concerned with the sides of the distribution rather than the heaviness of tails.

## A formal definition of fat tails {.unnumbered} 

The formal definition of fat tails comes from regular variation
Regular variation. A random variable, X, with distribution F has fat
tails if it varies regularly at infinity; that is there exists a positive
constant \tau > 0 such that:
$$\lim_{t \to \infty} \frac{1-F(tx)}{1-F(t)} =x^{-t},x>0$$

## Tail distributions {.unnumbered} 

* In the fat-tailed case, the tail distribution is Frechet:
$$H(x) = e^{−x^{−\tau}}$$

Lemma. A random variable X has regular variation at infinity (i.e. has fat tails) if and only if its distribution function F satisfies the following condition:

$$1 − F(x) = P(X > x) = Ax^{−\tau} + o(x^{−\tau})$$
for positive constant A, when $x \rightarrow \infty$.

* The expression $o(x−\tau$) is the remainder term of the Taylor-expansion of $P(X > x)$, it consists of terms of the type $Cx^{−\tau}$ for constant C and $j > \tau$
* As $x \rightarrow \infty$ the tails are asymptotically Pareto distributed:
$$F(x) \approx 1 - Ax^{-\tau}$$

The followng figure presents normal and fat tail distribution with respect to $\tau =2,4,6$

## Normal and fat distributions {.unnumbered} 

* The definition demonstrates that fat tails are defined by how rapidly the tails of the distribution decline as we approach infinity
* As the tails become thicker, we detect increasingly large observations that impact the calculation of moments:
$$E(X^m) = \int x^m f(x) \,dx$$
* If $E(X^m)$ exists for all positive $m$, such as for the normal distribution, the definition of regular variation implies that moments $m \geq 1$ are not defined for fat-tailed data.

# EVT in practice

There are two main approaches

1. Block Maxima
2. Peaks over thresholds (POT)

## Block maxima approach {.unnumbered} 

* This approach follows directly from the regular variation definition
where we estimate the GEV by dividing the sample into blocks and using the maxima in each block
* The procedure is rather wasteful of data and a relatively large sample is needed for accurate estimate for estimation (only the maxima of large blocks are used).

## Peaks over thresholds approach {.unnumbered} 

* This approach is based on models for all large observations that exceed a high threshold and hence makes better use of data on extreme values.
* There are two common approaches to POT
1. Fully parametric models (e.g. the Generalized Pareto distribution or
GPD)
2. Semi-parametric models (e.g. the Hill estimator)

### Generalized Pareto distribution {.unnumbered} 

* Consider a random variable $X$, fix a threshold $u$ and focus on the positive part of $X − u$
* The distribution $F_u(x)$ ( called excess distribution over u) is defined as follows

$$F_u(x) := P(X-u \leq |X> u)=\frac{F(x+u)-f(u)}{1-F(u)}$$

Mean excess function. 

If $\mathbb{E}(|X|) < \infty$ the mean excess function is
defined by
$$e(u) := E(X − u | X > u)$$

* Interpretation: $F_u$ is the distribution of the excess loss $X − u$ over $u$,
given that $X > u$. $e(u)$ is the mean of $F_u$ as a function of $u$.
* For continuous $X \sim F$ with $\mathbb{E}(X) < \infty$ the following formula holds:
$$ES_{\alpha} = e(VaR_{\alpha}(X)) + VaR_{\alpha}(X)$$

* Note that if $u$ is $VaR$ of $X$ then $F_u(x)$ is the probability that we exceed $VaR$ by a particular amount (a shortfall) given that $VaR$ is violated.
* The key result is that as $u \rightarrow \infty$, $F_u(x)$ converges to Generalized Pareto distribution (GPD), say $G_{\xi,\beta}(x)$
* $$G_{\xi,\beta}(x)=\begin{equation}
    \begin{cases}
      1-(1+\xi \frac{x}{\beta})^{\frac{1}{\xi}} & \xi \neq 0 \\
      1-e^{\frac{x}{\beta}}, & \xi = 0
    \end{cases}\,
\end{equation}$$
where $\beta > 0$ is the scale parameter, $\xi$ is known as shape. $x \geq 0$ when $\xi ≥ 0$ and $0 \leq x \leq −\frac{\beta}{\xi}$ when ξ < 0.
* Therefore we need to estimate both shape $\xi$ and scale $\beta$
parameters when applying GDP
* Recall, for certain values of $\xi$ the shape parameters, $G_{\xi,\beta}(x)$ becomes one of the three distributions (Frechet ($\xi > 0$); Weibull ($\xi < 0$), and Gumbel ($\xi = 0$)).

* Assume that $F_u(x) = G_{\xi,\beta}(x), \xi \ 0$, and some $u$.
* We obtain the following GPD-based formula for tail probabilities:

$$\begin{align*}
\bar F(x) &= \mathbb{P}(X>x) \\
&= \mathbb{P}(X>u) \mathbb{P}(X>x|X>u) \\
&= \bar F(u) \bar F_u(x-u) \\
&= \bar F(u) \bar F_u(x-u) \\
&= \bar F(u) \left(1+ \xi \frac{x-u}{\beta} \right)^{-\frac{1}{\xi}},x>u
\end{align*}$$

* Assuming we know $$ \bar F(u)$$, inverting this formula for $\alpha \geq  F(u)$ leads to

$$\begin{align*}
VaR_{\alpha}(X) &= F^{\leftarrow}(\alpha) \\
&= u+ \frac{\beta}{\xi} \left[ \frac{(1-\alpha)^{-\xi}}{\bar F(u)} -1 \right]
\end{align*}$$
and 

$$\begin{align*}
ES_{\alpha}(X) = \frac{VaR_{\alpha}(X)}{1-\xi} + \frac{\beta-\xi u}{1-\xi}, \xi<1
\end{align*}$$

### Hill method {.unnumbered} 

We have the approximation
$$\begin{align*}
F(x) = 1 − Ax^{−\tau}
\end{align*}$$
The tail index $−\tau$ can be approximated by the Hill method given as
$$\begin{align*}
\hat \xi = \frac{1}{\hat \tau} = \frac{1}{C_T} \sum_{i=1}^{C_T} \log \frac{x_{(i)}}{u}
\end{align*}$$
where $C_T$ is the number of observations in the tail, $2 \leq C_T \leq T$, as $T \rightarrrow \infty$ then $C_T \rightarrow \infty$, and $\frac{CT}{T} \rightarrow 0$, the notation $x_(i)$ indicates sorted data, where the maxima is denoted by x(1), and the second-largest observation by $x_(2)$.

<h4>Which method to choose?</h4>

* GPD, as the name suggests, is more general and can be applied
to all three types of tails
* Hill method on the other hand is in the maximum domain of
attraction (MDA) of the Frechet distribution
* Hence Hill method is only valid for fat-tailed data

<h4>Hill-based tail and risk measure estimates</h4>

* We have $\bar F(x)=1-F(x) \approx Ax^{-\tau},x \geq u$. Estimate $\tau$ and $\hat \tau$, and uu by $x_{(C_T)}$. (for $C_T$ sufficiently small)
* Note that $A=u \bar F(u)$, hence
$\hat A = x_{(C_T)}^{\hat \tau} \hat F(x_{(C_T)}^{\hat \tau}) \approx x_{(C_T)}^{\hat \tau} \frac{C_T}{T}$. So

$$\hat F(x) = \frac{C_T}{T}\left( \frac{x}{x_{(C_T)}} \right)^{-\hat \tau}$$
then
$$\widehat {VaR}_{\alpha} = \left( \frac{T}{C_T}(1-\alpha) \right)^{-\frac{1}{\tau}} x_{(C_T)}$$
and
$$\widehat {ES_\alpha} = \frac{\hat \tau}{\hat \tau-1} \widehat{VaR_{\alpha}}$$

# Homework {.unnumbered}

## Problem 1 {.unnumbered}

A Lebesgue$-$measurable function
$L:(0,\infty)\rightarrow\mathbb{R}$ is called **slowly varying at $\infty$** if
$$\lim_{x\rightarrow\infty}\frac{L(tx)}{L(x)}=1,\forall t>0$$ and is
called **regularly varying at $\infty$** if there is an
$\alpha\in\mathbb{R}$ such that
$$\lim_{x\rightarrow\infty}\frac{L(tx)}{L(x)}=t^{\alpha},\forall t>0.$$
Show that the following functions are slowly varying at $\infty:$

### Question a {.unnumbered} 

$L(x)=2+\cos(1/x),x>0.$

>For any $t>0,$
    $$\begin{align*}
    \lim_{x\rightarrow\infty}\frac{L(tx)}{L(x)} 
    &=\lim_{x\rightarrow\infty}\frac{2+\cos(1/tx)}{2+\cos(1/x)} \\
    &=1.
    \end{align*}$$
    
### Question b {.unnumbered} 

$L(x)=\ln(x),x>0.$

Show that the following functions are regularly varying at $\infty:$

>For any $t>0,$ by L'Hospital rule 
    $$\begin{align*}
            \lim_{x\rightarrow\infty}\frac{L(tx)}{L(x)} &= \lim_{x\rightarrow\infty}\frac{\ln(tx)}{\ln(x)} \\ 
            &=\lim_{x\rightarrow\infty}\left(\left(\frac{d}{dx}\ln(tx)\right):\left(\frac{d}{dx}\ln(x)\right)\right)\\
            &= \lim_{x\rightarrow\infty}\left(\left(\frac{1}{tx}\cdot t\right):\frac{1}{x}\right) \\
            &=1.
        \end{align*}$$
        
### Question c {.unnumbered} 

$h(x)=x^{-\theta},x,\theta>0.$

>For any $t>0,$
    $$\begin{align*}
    \lim_{x\rightarrow\infty}\frac{h(tx)}{h(x)} 
    &=\lim_{x\rightarrow\infty}\frac{(tx)^{-\theta}}{x^{-\theta}} \\
    &=\lim_{x\rightarrow\infty}t^{-\theta} \\
    &=t^{-\theta}=t^{\alpha}
    \end{align*}$$
where $\alpha=-\theta.$

### Question d {.unnumbered} 

$h(x)=(1+x)^{-\theta},x>-1,\theta>0.$

>For any $t>0,$ by L'Hospital rule
    $$\begin{align*}
    \lim_{x\rightarrow\infty}\frac{1+x}{1+tx} 
    &=\lim_{x\rightarrow\infty}\left(\left(\frac{d}{dx}(1+x)\right):\left(\frac{d}{dx}(1+tx)\right)\right) \\
    &=\lim_{x\rightarrow\infty}\frac{1}{t} \\
    &=\frac{1}{t},
    \end{align*}$$
    implying that
    $$\begin{align*}
    \lim_{x\rightarrow\infty}\frac{L(tx)}{L(x)} 
    &=\lim_{x\rightarrow\infty}\frac{(1+tx)^{-\theta}}{(1+x)^{-\theta}} \\
    &=\left(\lim_{x\rightarrow\infty}\frac{1+x}{1+tx}\right)^{\theta} \\
    &=t^{-\theta}=t^{\alpha}
    \end{align*}$$
    where $\alpha=-\theta.$

## Problem 2 {.unnumbered} 

For $\xi<0$ the generalized extreme value (GEV)
distribution takes the form
$$\begin{align*}
H_{\xi}(x)=\left\{\begin{matrix}\exp(-(1+\xi\cdot x)^{-1/\xi}) & x<-1/\xi\\1 & x\geq-1/\xi\end{matrix}\right.
\end{align*}$$
It is referred to as a Weibull distribution, although differing from the
standard Weibull distribution used in statistics and actuarial science.
Assume that $X$ has distribution $H_{\xi}$ and $Y=1+\xi\cdot X.$

### Question a {.unnumbered} 

Derive the distribution function of Y and its domain.

>The distribution function of $Y$ is 
$$\begin{align*}
            F_Y(x)
            &= \mathbb{P}(Y\leq x) \\
            &= \mathbb{P}(1+\xi\cdot X\leq x) \\
            &=\mathbb{P}\left(X\geq\frac{x-1}{\xi}\right) \\
            &=1-H_{\xi}\left(\frac{x-1}{\xi}\right)\\
            &= \left\{\begin{matrix}1-\exp(-(1+\xi\cdot(x-1)/\xi)^{-1/\xi}) & (x-1)/\xi<-1/\xi\\0 & (x-1)/\xi\geq-1/\xi\end{matrix}\right.\\
            &= \left\{\begin{matrix}0 & x\leq0\\1-\exp(-x^{-1/\xi}) & x>0\end{matrix}\right..
        \end{align*}$$

### Question b {.unnumbered} 

Verify that $Y$ has a standard Weibull distribution with density
    $$f_Y(y)=c\gamma y^{\gamma-1}\exp(-cy^{\gamma}),\forall y>0$$ where
    $c,\gamma>0$ are parameters (to be determined) in terms of $\xi.$
    
>The density of $Y$ is given by
    $$\begin{align*}
    f_Y(y) &=\frac{d}{dy}F_Y(y) \\
    &=(-1)^2\cdot\frac{-1}{\xi}\cdot y^{-1/\xi-1}\cdot\exp(-y^{-1/\xi}) \\
    &=c\gamma y^{\gamma-1}\exp(-cy^{\gamma})
    \end{align*}$$
    where
    $$\left\{\begin{matrix}c\gamma y^{\gamma-1}=-y^{-1/\xi-1}/\xi\\-cy^{\gamma}=-y^{-1/\xi}\end{matrix}\right. \\ \Rightarrow\gamma=-\frac{1}{\xi},c=1.$$

## Problem 3  {.unnumbered} 

Let $X$ be a random variable with the excess distribution
$$\begin{align*}
F(x)&=\mathbb{P}(X-u\leq x|X>u) \\
&=1-c(k+x)^{-\theta},\forall x\geq u
\end{align*}$$
where $c,\theta,u>0$ and $k\in\mathbb{R}.$ Let $\alpha_1,\alpha_2$
satisfy $F(u)<\alpha_1\leq\alpha_2<1.$

>Assume that $X$ has the excess tail distribution
$\overline{F}(x)=1-F(x)=c(k+x)^{-\theta}.$ Since $F$ is continuous,
$$\mathbb{P}(X\leq VaR_{\alpha}(X))=\alpha\hspace{4mm}\text{and}\hspace{4mm}ES_{\alpha}(X)=\frac{1}{1-\alpha}\int_{\alpha}^1VaR_{p}(X)dp,\forall\alpha\in[0,1].$$

### Question a {.unnumbered} 

Show that
$$\begin{equation}
  VaR_{\alpha_2}(X)=\left(\frac{1-\alpha_1}{1-\alpha_2}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k
\end{equation}$$
where
    $VaR_{\alpha}(X)=\inf\left\{x:F(x)\geq\alpha\right\},\forall\alpha\in[0,1].$

>The equalities
    $$\begin{align*}
    \left(\frac{k+VaR_{\alpha_2}(X)}{k+VaR_{\alpha_1}(X)}\right)^{\theta} 
    &=\frac{c(k+VaR_{\alpha_1}(X))^{-\theta}}{c(k+VaR_{\alpha_2}(X))^{-\theta}}\\
    &=\frac{\overline{F}(VaR_{\alpha_1}(X))}{\overline{F}(VaR_{\alpha_2}(X))}\\
    &=\frac{1-\alpha_1}{1-\alpha_2}
    \end{align*}$$
    clearly imply $$\begin{equation}
  VaR_{\alpha_2}(X)=\left(\frac{1-\alpha_1}{1-\alpha_2}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k
\end{equation}.$$

### Question b {.unnumbered} 

Show that if $\theta>1,$ then
    $$ES_{\alpha_2}(X)=\frac{\theta}{\theta-1}\cdot\left(\frac{1-\alpha_1}{1-\alpha_2}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k$$
    where
    $ES_{\alpha}(X)=\mathbb{E}(X|X\geq VaR_{\alpha}(X)),\forall\alpha\in[0,1].$

>By part (a),
    $$VaR_{p}(X)=\left(\frac{1-\alpha_1}{1-p}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k$$
    and hence $$\begin{aligned}
            ES_{\alpha_2}(X)
            &= \frac{1}{1-\alpha_2}\int_{\alpha_2}^1VaR_{p}(X)dp \\
            &=\frac{1}{1-\alpha_2}\int_{\alpha_2}^1\left[\left(\frac{1-\alpha_1}{1-p}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k\right]dp\\
            &= \frac{(1-\alpha_1)^{1/\theta}}{1-\alpha_2}(k+VaR_{\alpha_1}(X))\int_{\alpha_2}^1\left(\frac{1}{1-p}\right)^{1/\theta}dp-\frac{1}{1-\alpha_2}\cdot\int_{\alpha_2}^1kdp\\
            &= \frac{(1-\alpha_1)^{1/\theta}}{1-\alpha_2}(k+VaR_{\alpha_1}(X))\cdot\frac{\theta}{\theta-1}\cdot\frac{1}{(1-\alpha_2)^{1/\theta-1}}-k\\
            &= \frac{\theta}{\theta-1}\cdot\left(\frac{1-\alpha_1}{1-\alpha_2}\right)^{1/\theta}(k+VaR_{\alpha_1}(X))-k
        \end{aligned}$$ as desired.
