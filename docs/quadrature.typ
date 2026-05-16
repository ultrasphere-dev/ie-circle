#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)
#let definition = thmplain("definition", "Definition")
#let theorem = thmplain("theorem", "Theorem")
#let lemma = thmplain("lemma", "Lemma")
#let proof = thmproof("proof", "Proof")
#let span = $op("span")$
#let sign = $op("sign")$
#let ts = $t_"start"$

== Integrals involving Fourier basis functions

The below thorem is used everywhere.

#theorem[Fourier Integral][
  $
    integral_0^(2 pi) e^(i m x) dd(x) = cases(2 pi &(m = 0), 0 &(m in ZZ without {0}))
  $
] <fourier-integral>
#proof[
  $
    integral_0^(2 pi) e^(i m x) dd(x) & = (e^(i m dot 2 pi) - e^(i m dot 0))/(i m) \
    & = cases(2 pi &(m = 0), 0 &(m in ZZ without {0}))
  $
]

- Integral equations coming from common partial differential equations in $SS^1$ ($RR^2$) often involve singular integrals, derivative of integrals.
- To compute these integrals, exact values of integrals of Fourier basis functions are needed.
- To deal with the singularity while maintaining $2 pi$-periodicity, we need a function which only has singularity at $0 (2 pi)$ and is $2 pi$-periodic.
  - $csc^n t, cot^n t, n in NN$: singularities at $pi$ as well
  - $csc^n t/2, n in.not 2 NN$: not $2 pi$-periodic
  - $cot^n t/2, n in NN$: ok
  - $csc^(2 n) t/2, n in 2 NN$: ok

#definition[Cauchy Principle value on a circle][
  $forall alpha > 0. forall t in [0, 2 pi). forall f in C^(0, alpha). (0, 2 pi)$
  $
    integral.dash_0^(2 pi) f(x) cot((x-t)/2) dd(x) := lim_(epsilon -> 0) (integral_(epsilon)^(2 pi - epsilon) f(x + t) cot(x/2) dd(x))
  $
]
#definition[Hadamard Finite-Part on a circle][
  $forall alpha > 0. forall t in [0, 2 pi). forall n in NN without {1}. forall f in C^(n-1,alpha) (0, 2 pi).$
  $
    integral.dash_0^(2 pi) f(x) cot^n ((x-t)/2) dd(x)
    &:= lim_(epsilon -> 0) F_0(epsilon)
  $
  where
  $
    forall epsilon in (0, pi). integral_(epsilon)^(2 pi - epsilon) f(x + t) cot^n (x/2) dd(x) = F_0 (epsilon) + F_1 (epsilon) \
    lim_(epsilon -> 0) F_0 (epsilon) < infinity \
    lim_(epsilon -> 0) log(4 sin^2 (epsilon/2)) 1/(F_1 (epsilon)) < infinity
  $
]
#theorem[Hadamard Finite-Part on a circle][
  $forall n in NN. forall f in C^(n, alpha) (RR\/2 pi) inter L^(1 +) (RR\/2 pi).$Let
  $
    (T_n f) (t) & := integral.dash_0^(2 pi) f(x) cot^n ((x-t)/2) dd(x)
  $
  then
  $
    (T_(n + 1) f) (t) & := -2/n dv(, t) (T_(n) f) (t) - (T_(n - 1) f) (t)
  $
  where
  $
    (T_1 f) (t) & := integral.dash_0^(2 pi) f(x) cot ((x - t)/2) dd(x) \
    (T_0 f) (t) & := integral_0^(2 pi) f(x) dd(x)
  $
] <hadamard-recurrence>
#theorem[
  $forall n in NN. forall f in C^(n, alpha) (RR\/2 pi) inter L^(1 +) (RR\/2 pi).$
  $
    dv(, t) integral.dash_0^(2 pi) f(x) cot^n ((x - t)/2) dd(x) = integral.dash_0^(2 pi) f'(x) cot^n ((x - t)/2) dd(x)
  $
] <derivative-of-fp>
#proof[
  $
    dv(, t) integral.dash_0^(2 pi) f(x) cot^n ((x - t)/2) dd(x)
    &= dv(, t) integral.dash_0^(2 pi) f(x + t) cot^n (x/2) dd(x) \
    &= dv(, t) lim_(epsilon -> 0) F(t, epsilon) \
    &=_? lim_(epsilon -> 0) dv(, t) F(t, epsilon) \
    &= integral.dash_0^(2 pi) f'(x) cot^n ((x - t)/2) dd(x)
  $
]
#theorem[
  $
    integral.dash_0^(2 pi) f'(t) g(t) dd(t) =_? - integral.dash_0^(2 pi) f(t) g'(t) dd(t)
  $
]
#theorem[
  Let
  $
    I_(m,n) = integral.dash_0^(2 pi) e^(i m t) cot^n (t/2) dif t
  $
  then following recurrence relation holds:
  $
    forall m in ZZ. forall n in NN without {1}. I_(m,n) &= (2 i m)/(n-1) I_(m,n-1) - I_(m,n-2)
  $
  with initial values:
  $
    I_(m,0) & = 0,    & quad I_(m,1) & = 2 pi i sgn(m) quad (m != 0) \
    I_(0,0) & = 2 pi, & quad I_(0,1) & = 0
  $
]
#proof[
  - $I_(m, 0), m in ZZ$: @fourier-integral.
  - $I_(0, 1) = 0$: Due to the asymmetry of integrand $cot t/2$.
  - $I_(m, 1), m in ZZ$: Follows #cite(<kress_linear_2014>, supplement: [Lemma 8.23.]).
    - $cot t/2 = (cos t/2)/(sin t/2) = i (e^((i t)/2) + e^(- (i t)/2))/(e^((i t)/2) - e^(- (i t)/2)) = i (e^(i t) + 1)/(e^(i t) - 1)$
    - $therefore forall m in NN. (e^(i m t) - 1) cot (t/2) = (e^(i t) - 1) (sum_(j = 0)^(m - 1) e^(i j t)) cot (t/2) = i (e^(i t) + 1) sum_(j = 0)^(m - 1) e^(i j t)$.
    - $therefore forall m in NN. I_(m,1) - I_(0, 1) = 2 pi i because$Integrating both hands and @fourier-integral
    - $therefore forall m in NN. I_(m,1) = 2 pi i$
    - $forall m in NN, I_(-m,1) = overline(I_(m,1)) = - 2 pi i$
  - Recurrence relation: Due to @hadamard-recurrence @derivative-of-fp and $(e^(i m t))' = i m e^(i m t)$.
]
#theorem[
  Let
  $
    J_(m,n) &= integral.dash_0^(2 pi) e^(i m t) log(4 sin^2 (t/2)) cot^n (t/2) dif t
  $
  then following recurrence relation holds:
  $
    forall m in ZZ. forall n in NN without {1}. J_(m,n) &= (2 i m)/(n-1) J_(m,n-1) - J_(m,n-2) + 2/(n-1) I_(m,n)
  $
  with initial values:
  $
    J_(m,0) &= -(2 pi)/abs(m), &quad J_(m,1) &= 2 pi i sgn(m) ( 2 H_(abs(m)) - 1/abs(m) ), H_m := sum_(k = 1)^m 1/k quad (m != 0) \
    J_(0,0) &= 0, &quad J_(0,1) &= 0
  $
]
#proof[
  - $J_(m, 0), m in ZZ$: Follows #cite(<kress_linear_2014>, supplement: [Lemma 8.23.]).
  - $J_(0, 1)$: Due to the asymmetry of integrand $log(4 sin^2 (t/2)) cot (t/2)$.
  - $J_(m, 1), m in ZZ without {0}$:
    - $L(t) := log(4 sin^2 (t/2)) = - sum_(k in ZZ without {0}) e^(i k t)/abs(k)$
    - $forall m in NN. J_(m, 1) = integral.dash_0^(2 pi) e^(i m t) L(t) L'(t) dd(t) = - (i m)/2 integral_0^(2 pi) e^(i m t) L^2 (t) dd(t) = - (i m)/2 integral_0^(2 pi) e^(i m t) (sum_(k in ZZ without {0}) e^(i k t)/abs(k)) (sum_(l in ZZ without {0}) e^(i l t)/abs(l)) dd(t)
      = (2 pi) sum_(k, l in ZZ without {0}, k + l + m = 0) 1/(abs(k) abs(l)) = (2 pi) sum_(k in ZZ without {0}, l in ZZ, k + l + m = 0) 1/(abs(k) abs(l)) = (2 pi) sum_(k in ZZ without {0, -m}) 1/(abs(k) abs(m + k))$
    - $forall m in NN. sum_(k in ZZ without {0, -m}) 1/(abs(k) abs(m + k)) = sum_(k = 1)^(infinity) 1/(k (m + k)) + sum_(k = -m + 1)^(-1) 1/(abs(k) abs(m + k)) + sum_(k = -infinity)^(- m + 1) 1/(abs(k) abs(m + k)) = sum_(k = 1)^(infinity) 1/(k (m + k)) + sum_(k = 1)^(m - 1) 1/(k (m - k)) + sum_(k = 1)^(infinity) 1/(k (m + k)) = 2/m sum_(k = 1)^(infinity) (1/k - 1/(m + k)) + 1/m sum_(k = 1)^(m - 1) (1/k + 1/(m - k)) = 2/m (sum_(k = 1)^m 1/k + sum_(k = 1)^(m - 1) 1/k) = 2/m (2 H_m - 1/m)$
]
#theorem[Estimates for $I_(m,n), J_(m,n)$][
  + $forall m in ZZ without {0}. forall n in NN_0. abs(I_(m,n)) <= 2 pi n abs(m)^(n-1)$
  + $forall m in ZZ without {0}. forall n in NN_0. abs(J_(m,n)) <= 2 pi (n + 1)^2 abs(m)^(n)$
]
#proof[
  Proved by induction for $n$
  + $I_(m,n)$
    - $n = 0$: $abs(I_(m, 0)) = 0$
    - $n = 1$: $abs(I_(m, 1)) = 2 pi$
    - $n > 1$: $abs(I_(m, n)) = abs((2 i m)/(n-1) I_(m, n-1) - I_(m, n-2)) <= (2 abs(m))/(n - 1) abs(I_(m,n-1)) + abs(I_(m,n-2)) <= (2 abs(m))/(n - 1) dot 2 pi (n - 1) abs(m)^(n - 2) + 2 pi (n - 2) abs(m)^(n - 3) <= 4 pi abs(m)^(n - 1) + 2 pi (n - 2) abs(m)^(n - 3) <= 2 pi n abs(m)^(n - 1)$
  + $J_(m,n)$
    - $n = 0$: $abs(J_(m, 0)) = (2 pi)/abs(m) <= 2 pi$
    - $n = 1$: $abs(J_(m, 1)) = 2 pi abs(2 H_(abs(m)) - 1/abs(m)) <= 8 pi abs(m)$
    - $n > 1$: $abs(J_(m,n)) = abs((2 i m)/(n-1) J_(m,n-1) - J_(m,n-2) + 2/(n-1) I_(m,n)) <= (2 abs(m))/(n - 1) abs(J_(m, n-1)) + abs(J_(m, n-2)) + 2/(n - 1) abs(I_(m, n)) <= (2 abs(m))/(n - 1) dot 4 pi n^2 abs(m)^(n - 1) + 4 pi (n-1)^2 abs(m)^(n - 2) + 4 pi n/(n - 1) abs(m)^(n - 1) <= 4 pi (4 n + (n - 1)^2 + 2) abs(m)^n = 4 pi (n^2 + 2n + 1) abs(m)^n = 4 pi (n + 1)^2 abs(m)^n$
]
// - $n = 2$: $<= 4 pi (2 (n - 1) + 1 + 2) abs(m)^(n - 1) = 4 pi 5 abs(m)^(n - 1) <= 4 pi n^2 abs(m)^n$
// - $n > 2$:$
== Subspace $U_N$

#lemma[
  $forall N' in NN. t_j := (2 pi j)/N'. forall m in ZZ. forall ts in [0, 2 pi).$
  $
    (2 pi)/N' sum_(j=0)^(N'-1) e^(i m (t_j + ts)) = cases(2 pi &(m equiv 0 mod N'), 0 &("otherwise"))
  $
] <fourier-sum>
#definition[
  $U_N := span({e^(i m x) | m in ZZ, abs(m) < N})$
]
#theorem[Trapezoidal Rule for $U_N$][
  $forall N in NN. N$-point trapezoidal rule is exact for $U_N$.
] <trapezoidal-rule>
#proof[
  @fourier-sum
]
#lemma[Trapezoidal Rule for inner product for $U_N$][
  Let $N' := dim U_N = 2 N - 1$.
  Let $t_j := (2 pi j)/N'$ for $j = 0, ..., N' - 1$.
  $forall f, g in U_N.$
  $
    integral_0^(2 pi) f(t) g(t) dd(t) = (2 pi)/N' sum_(j=0)^(N'-1) f(t_j) g(t_j)
  $
] <dft-trapezoidal>
#proof[
  Since, $f(t) g(t) in U_(2 N - 1)$ @trapezoidal-rule can be applied with $N' = 2 N - 1$ points.
]

== Quadratures

Now we consider the way to express discrete Fourier expansion of a function in $U_N$ using the values of the function at $ts, ts + (2 pi)/N', ts + (4 pi)/N', ...$.

#lemma[
  $forall N in NN. forall f in U_N. N := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    f(x) = sum_(j = 0)^(N'-1) f(t_j + ts) dot (1/N' sum_(abs(m) < N) e^(- i m (t_j + ts)) e^(i m x))
  $
] <dft>
#proof[
  $
    f(x) & = sum_(abs(m) < N) integral_0^(2 pi) f(t) e^(-i m t)/sqrt(2 pi) dd(t) dot e^(i m x)/sqrt(2 pi) \
    & =_(because #ref(<dft-trapezoidal>)) 1/(2 pi) sum_(abs(m) < N) (2 pi)/N' sum_(j=0)^(N'-1) f(t_j + ts) e^(-i m (t_j + t)) dot e^(i m x) \
    & = sum_(j = 0)^(N'-1) f(t_j + ts) dot (1/N' sum_(abs(m) < N) e^(- i m (t_j + ts)) e^(i m x))
  $
]

#definition[Lagrange basis][
  $forall N in NN. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    L^ts_j (x) := 1/N' sum_(abs(m) < N) e^(- i m (t_j + ts)) e^(i m x)
  $
]

#let ip(x, y) = $lr(chevron.l #x, #y chevron.r)$
#theorem[
  $forall N in NN. N' := 2 N - 1. forall ts, ts' in [0, 2 pi). forall i, j in {0, ..., N' - 1}. t_i := (2 pi i)/N'. t_j := (2 pi j)/N'.$
  $
    ip(L^ts_i, L^(ts')_j) = (2 pi)/N'^2 sum_(abs(m) < N) e^(i m (t_j + ts' - t_i - ts))
  $
]
#proof[
  $
    ip(L^ts_i, L^(ts')_j) &=
    integral_0^(2 pi) L^ts_i (t) overline(L^(ts')_j (t)) dd(t) \
    &= integral_0^(2 pi) (1/N' sum_(abs(m) < N) e^(- i m (t_i + ts)) e^(i m t)) (1/N' sum_(abs(n) < N) e^(i n (t_j + ts')) e^(- i n t)) dd(t) \
    &= 1/(N'^2) sum_(abs(m) < N) sum_(abs(n) < N) e^(- i m (t_i + ts)) e^(i n (t_j + ts')) integral_0^(2 pi) e^(i (m - n) t) dd(t) \
    &= 1/(N'^2) sum_(abs(m) < N) sum_(abs(n) < N) e^(- i m (t_i + ts)) e^(i n (t_j + ts')) delta_(m, n) 2 pi \
    &=(2 pi)/(N'^2) sum_(abs(m) < N) e^(i m (t_j + ts' - t_i - ts)) \
  $
]

Combining the above lemma with the exact integral values of Fourier basis functions, we can exactly compute the integrals of functions in $U_N$ multiplied by $cot^n (t/2)$ or $log(4 sin^2 (t/2)) cot^n (t/2)$, by using only the values of the function at $ts, ts + (2 pi)/N', ts + (4 pi)/N', ...$.

#theorem[Generalized Garrick--Wittich quadrature for $U_N$][
  $forall n in NN_0. forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) f(t) cot^n (t/2) dd(t) = sum_(j=0)^(N'-1) f(t_j + ts) P_j^(N',n,ts)
  $
  $
    P_j^(N',n,ts) := 1/N' sum_(abs(m) < N) I_(m,n) e^(- i m (t_j + ts))
  $
]
#theorem[Generalized Kussmaul--Martensen (Kress) quadrature for $U_N$][
  $forall n in NN_0. forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) f(t) log(4 sin^2 (t/2)) cot^n (t/2) dd(t) = sum_(j=0)^(N'-1) f(t_j + ts) Q_j^(N',n,ts)
  $
  $
    Q_j^(N',n,ts) := 1/N' sum_(abs(m) < N) J_(m,n) e^(- i m (t_j + ts))
  $
]

== Error estimates

= Integral Equations

#lemma[
  $forall n in NN. forall N in NN. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) K(x, y) cot^n ((x - y)/2) dd(y) approx (-1)^n sum_(j = 0)^(N'-1) K(x, t_j + ts) P_j^(N',n,(ts - x))
  $
]
#proof[
  $
    integral.dash_0^(2 pi) K(x, y) cot^n ((x - y)/2) dd(y)
    &= (-1)^n integral.dash_0^(2 pi) K(x, y) cot^n ((y - x)/2) dd(y) \
    &= (-1)^n integral.dash_0^(2 pi) K(x, t + x) cot^n (t/2) dd(t) \
    &approx (-1)^n sum_(j = 0)^(N'-1) K(x, t_j + ts) P_j^(N',n,(ts - x)) \
  $
]

#lemma[
  $forall n in NN. forall N in NN. N' := 2 N - 1. forall j in NN_0. t_j := (2 pi j)/N'. forall ts, ts' in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) K(t_i + ts, y) phi(y) cot^n ((t_i + ts - y)/2) dd(y) approx (-1)^n sum_(j = 0)^(N'-1) K(t_i + ts, t_j + ts') phi(t_j + ts') P_(j-i)^(N',n,(ts' - ts))
  $
]

#let binv = $B^("inv")$
#theorem[
  $X$: Banach space,
  $forall A in B(X). forall T in binv(X). norm(T^(-1) (T - A)) < 1 ==> A in binv(X)$
]
#proof[
  Neumann series implies $(I - (T^(-1) (A - T))) in binv(X)$ and
  $
    norm(A^(-1)) = norm((I - (T^(-1) (T - A)))^(-1)) <= 1/(1 - norm(T^(-1) (T - A)))
  $
]

#definition[
  $
    I_N: -> U_N, f |-> sum_(j = 0)^(N'-1) f(t_j + ts) L_ts (x)
  $
]

#let dphi = $tilde(phi)^((N))$
#theorem[
  Let $phi$ solution of
  $
    phi(s) + sum_(n = 0)^M integral.dash_0^(2 pi) K_n (s, t) cot^n ((s - t)/2) phi(t) dd(t) \
    + sum_(n = 0)^M integral.dash_0^(2 pi) L_n (s, t) log(4 sin^2 ((s - t)/2)) cot^n ((s - t)/2) phi(t) dd(t) = f(s)
  $ <ie-original>
  and $dphi in U_N$ solution of
  $
    dphi(s) + I_N sum_(n = 0)^M integral.dash_0^(2 pi) K_n (s, t) cot^n ((s - t)/2) dphi(t) dd(t) \
    + I_N sum_(n = 0)^M integral.dash_0^(2 pi) L_n (s, t) log(4 sin^2 ((s - t)/2)) cot^n ((s - t)/2) dphi(t) dd(t) = I_N f(s)
  $ <ie-interpolated>
  assume that @ie-original is solvable. Then
  + $exists N_s in NN. norm(T - A) < norm(T^(-1))^(-1)$
  + $forall N_s ["satisfies above condition"]. forall N >= N_s.$@ie-interpolated has a unique solution
]

#theorem[
  Let ${dphi_j}_(j = 0)^(N'-1)$ solution of
  $
    sum_(j = 0)^(N'-1) L_j^(ts')(t_i + ts) dphi_j + sum_(n = 0)^M (-1)^n sum_(j=0)^(N'-1) K_n (t_i + ts, t_j + ts') P_j^(N',n,ts') dphi_j \
    + sum_(n = 0)^M (-1)^n sum_(j=0)^(N'-1) L_n (t_i + ts, t_j + ts') Q_j^(N',n,ts') dphi_j = f(t_i + ts)
  $
  Then
  $
    dphi (x) = sum_(j = 0)^(N'-1) dphi_j L^ts'_j (x)
  $
]
#proof[
  Evaluate @ie-interpolated at $s = t_i + ts$.
  $
    sum_(j = 0)^(N' - 1) L_j^(ts')(t_i + ts) dphi_j + sum_(n = 0)^M sum_(j=0)^(N'-1) K_n (t_i + ts, t_j + ts') P_(j-i)^(N',n,ts'-ts) dphi_j \
    + sum_(n = 0)^M (-1)^n sum_(j=0)^(N'-1) L_n (t_i + ts, t_j + ts') Q_(j-i)^(N',n,ts'-ts) dphi_j = f(t_i + ts)
  $
]

#bibliography("quadrature.bib")
