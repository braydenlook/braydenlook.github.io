---
layout: post
title:  "Thinking About Completeness (Part I (maybe))"
date:   2024-10-19 10:27:48 -0600
tags: [estimation, statistics, math-stats, functional-analysis]
---


Let $f(t\|\theta)$ be a family of pdfs or pmfs for a statistic $$T(X)$$. The family of probability distributions is called complete if $$\mathbb{E}_{\theta}[g(T)] = 0$$ for all $$\theta$$ implies that $$P_{\theta}(g(T) = 0) = 1$$ for all $$\theta$$. Equivalently, $T(X)$ is called a complete statistic. Sometimes we refer to the family of pdfs $f(t\|\theta)$ as complete, rather than the statistic.

This is the definition of completeness in the context of statistical estimation. After being given this definition in class, you're usually given a bunch of problems that prove whether or not a statistic is complete, and then ultimately use completeness to find uniformly minimum variance unbiased estimators (UMVUEs). But why is it defined this way? What is this really telling us? 

In short, completeness gives us the *uniqueness* part of UMVUEs, which is important for finding an estimator that is lowest variance. When a family of pdfs is complete, the expectation taken with respect to that pdf can be thought of as an injective map, meaning that we have a one-to-one correspondence between input and output.

### Linear Algebra Review

Suppose we have a matrix $A \in \mathbb{R}^{n \times m}$. This matrix can be thought of as a mapping $A: \mathbb{R}^m \rightarrow \mathbb{R}^n$ in the sense that if we given $A$ an input $x \in \mathbb{R}^m$, then we get an output $y \in \mathbb{R}^n$ such that $Ax = y$. 

Denote the $i^{th}$ row of $A$ as $A^i$. Then we can write out the matrix multiplication with $x$ as

$$
\begin{aligned}
	Ax = 
	\begin{bmatrix}
	a_{11} & a_{12} & ... & a_{1m}\\
	a_{21} & a_{22} & ... & a_{2m}\\
	... & ... & ... & ...\\
	a_{n_1} & a_{n2} & ... & a_{nm}
	\end{bmatrix}
	\begin{bmatrix}
	x_1\\
	x_2\\
	...\\
	x_m
	\end{bmatrix}
	=
	\begin{bmatrix}
	\langle A^1, x\rangle\\
	\langle A^2, x\rangle\\
	...\\
	\langle A^n, x\rangle\\
	\end{bmatrix}
	=
	\begin{bmatrix}
	y_1\\
	y_2\\
	...\\
	y_n
	\end{bmatrix}
\end{aligned}
$$

That is, every entry $y_i$ is an inner product between the $i^{th}$ row of $A$ and $x$. Note that in a way, $y$ can be thought of as a function of its index, $i$. Meaning that if you give me an index, say 4, I give you an output, $y_4 = \langle A^4, x\rangle$. 

When we have the property that $Ax = 0$ only when $x = 0$, we call the mapping $A$ *injective*. Or equivalently, $A$ is a tall ($n \ge m$) matrix with full rank. Injectivity means that we have a one-to-one mapping from $\mathbb{R}^m$ to $\mathbb{R}^n$, so that no two distinct vectors $x_1, x_2 \in \mathbb{R}^m$ can be mapped to the same vector $y \in \mathbb{R}^n$ by $A$. 

This property follows from the statement $Ax = 0$ only when $x = 0$ because if $Ax_1 = y$ and $Ax_2 = y$, then $Ax_1 - Ax_2 = A(x_1 - x_2) = 0$. I.e., if $A$ is not injective, then we can always find a non-zero vector $x \in \mathbb{R}^m$ for which $Ax = 0$ by just subtracting two vectors that map to the same output.
### Making the Connection

Now we consider a family of pdfs $\{f_\theta\}$ indexed by $\theta$. For example, we could say that $f_\theta$ represents a Normal distribution with variance 1 and mean $\theta$. So we have a set of pdfs, where each element of the set is just a unit-variance normal pdf with different means. 

Now let's think about taking the expectation of a function of some statistic $T$, where the pdf of $T$ is $f_\theta$. This is given by the expression

$$\mathbb{E}_\theta[g(T)] = \int_{t\in T} g(t) f_T(t;\theta)\, dt.$$

Note that if we treat $\theta$ as a free variable, $$\mathbb{E}_\theta[g(T)]$$ is a function of $\theta$. For example, if $f_\theta$ represents our example of a unit-variance normal distribution with mean $\theta$, then if we set $g(T) = T$, we have

$$
\mathbb{E}_\theta[T] = \int_{t\in T} t f_T(t;\theta)\, dt = \theta.
$$

For a different function $g$, we get a different function of $\theta$. We'll refer to this output function more generally as $h(\theta)$, so that we may say $\mathbb{E}_\theta[g(T)] = h(\theta)$. 


---


Oftentimes we think of the expectation as an inner product on a function space. There are a LOT of assumptions we're sweeping under the rug here, but one way to intuitively see this is that in a traditional linear algebra setting where we have vectors $a, b \in \mathbb{R}^n$, the inner product is defined as 

$$
\langle a, b\rangle := \sum_{i=1}^n a_i b_i.
$$

Well, since sums are just a discrete analog to integrals, what if we had two real *functions* $a, b$ and slapped an integral sign over the sum to say something like

$$
\langle a, b\rangle := \int a(t)b(t)\, dt
$$

Mathematicians everywhere are seething, tears run down their faces, they're ripping their hair out, pleading. "NOOOOOOO, YOU CAN'T JUST REPLACE A SUM WITH AN INTEGRAL AND SAY IT FOLLOWS FROM THE DISCRETE CASE!!" But Jesus has already cried out. The temple veil has been torn in two. The barrier has been broken. Cross the plane. Touchdown. Greg Jennings.

Anyway this now looks a lot like an expectation of $a$ taken with respect to $b$. 


---


If we go back to our scenario and fix $\theta$ to be some $\theta_0$, then we can write

$$
\langle g, f_{\theta_0} \rangle = \mathbb{E}_\theta[g(T)]\Big |_{\theta = \theta_0} = \int_{t\in T} g(t) f_T(t;\theta_0)\, dt = h(\theta_0).
$$

So for each given $\theta$, this expectation is an inner product between $g$ and $f_{\theta_0}$. Well this sounds awfully similar to our simple linear algebra scenario with $A$, $x$, and $y$. If we connect the dots we can think of $g$ as being analogous to $x$, $h$ to $y$, and $\mathbb{E}_\theta[\cdot]$ to $A$.

How does $\theta$ itself factor in? Well, remember when I mentioned that in the linear algebra case, $y$ could be thought of as a function of its index, $i$? Here, the index is $\theta$. So just as we could write:

$$
\begin{aligned}
	Ax =
	\begin{bmatrix}
	\langle A^1, x\rangle\\
	\langle A^2, x\rangle\\
	...\\
	\langle A^n, x\rangle\\
	\end{bmatrix}
	=
	\begin{bmatrix}
	y_1\\
	y_2\\
	...\\
	y_n
	\end{bmatrix}
	= y(i), \,\, i = 1, 2, ..., n
\end{aligned}
$$

We can analogously write something like

$$
\begin{aligned}
	\mathbb{E}_\theta[g(T)] =  
	\begin{bmatrix}
	\langle f_{\theta_1}, x\rangle\\
	\langle f_{\theta_2}, x\rangle\\
	\langle f_{\theta_3}, x\rangle\\
	...\\
	\end{bmatrix}
	=
	\begin{bmatrix}
	h(\theta_1)\\
	h(\theta_2)\\
	h(\theta_3)\\
	...\\
	\end{bmatrix}
	= h(\theta), \,\, \theta \in \Theta
\end{aligned}
$$

Except that $\theta$ is not discrete, but is now continuous.

Now let's return to the definition of completeness to try to tie this together:

Completeness says that $$\mathbb{E}_\theta[g(T)] = 0$$ for all $\theta$ implies that $g(T) = 0$ almost surely. This should now look analogous to the idea of injectivity--remember we said that $A$ is injective if $Ax = 0$ only when $x = 0$! This means that if $T$ is complete, and $$\mathbb{E}_{\theta}[g(T)] = h(\theta)$$ for all $\theta$, then $g(T)$ is the only function for which this is true. There is nothing else you could put into the expectation $$\mathbb{E}_{\theta}[\cdot]$$ such that it would equal $h(\theta)$ *for all $\theta.$* 

### Implications

So completeness is guaranteeing us a one-to-one correspondence between a function $g$ and another function $h$. Why is this useful in the context of estimation? Because it gets us from the Rao-Blackwell theorem to the Lehmann-Scheffe Theorem.

Formally, the Rao-Blackwell Theorem states:

> Let $$W$$ be any unbiased estimator of $$h(\theta)$$, and let $$T$$ be a sufficient statistic for $$\theta$$. Define $\phi(T) := \mathbb{E}[W\|T]$. Then $$\mathbb{E}_{\theta}[\phi(T)] = h(\theta)$$ and $$\text{Var}(\phi(T)) \le \text{Var}(W)$$ for all $$\theta$$. That is, $$\phi(T)$$ is a uniformly better unbiased estimator of $$h(\theta)$$.

Less formally, Rao-Blackwell theorem states that an estimator based on a sufficient statistic will have uniformly lower variance than an estimator based on a statistic that is not sufficient. So if we want the lowest variance possible for all possible $\theta$, we need a sufficient statistic.

So if we have an unbiased estimator $W$ that is based on a statistic $S$ that is not sufficient, we can improve it with a sufficient statistic $T$ by defining $g(T) = \mathbb{E}[W\|T]$. Now we have a better estimator. But what if there is another unbiased estimator $V$ and we defined $f(T) = \mathbb{E}[V\|T]$? How do we know if $g(T)$ or $f(T)$ is better? What if there are a thousand other unbiased estimators? How can we know if the estimator we chose is the best?

If $T$ is not complete, then we wouldn't know. But if $T$ is complete, then we get uniqueness from injectivity! In other words, if we started with $W$ and defined $g(T) := \mathbb{E}[W\|T]$ to get an unbiased estimator that is a function of a complete sufficient statistic, then there is no other function $f$ for which $$\mathbb{E}_{\theta}[f(T)] = h(\theta)$$ for all $\theta.$ 

How do we know this just from completeness? The same argument as in the linear algebra case. Suppose that $T$ is complete, and suppose such an $f$ exists.  Define $\tilde f(T) := f(T) - g(T)$. Then 

$$
\begin{aligned}
\mathbb{E}_{\theta}[\tilde f(T)] &= \mathbb{E}_\theta[f(T) - g(T)]\\[10pt]
&= \mathbb{E}[f(T)] - \mathbb{E}[g(T)]\\[10pt]
&= h(\theta) - h(\theta) = 0.
\end{aligned}
$$ 

But since $T$ is complete, we know there exists no function for which this is true. Thus, $f$ cannot exist, and so $g$ is unique. This is what the Lehmann-Scheffe theorem gives us:

> If T is a complete sufficient statistic for $\theta$ and $$\mathbb{E}[g(T)] = h(\theta)$$, then $g(T)$ is the UMVUE of $h(\theta).$

That's pretty neat. But the mathematicians still scream in the distance. They're wailing, telling us that we can't define an inner product without an inner product space. They're begging us to define the function space for which this mapping holds. 


