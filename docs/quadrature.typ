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
    integral_0^(2 pi) e^(i m x) dd(x) & = (e^(i m dot 2 pi) - e^(i m dot 0))/(i m)  \
                                   & = cases(2 pi &(m = 0), 0 &(m in ZZ without {0}))
  $
]

- Integral equations coming from common partial differential equations in $RR^2$ often involve singular integrals, derivative of integrals.
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
  $forall n in NN. forall f in C^(n, alpha) (RR\/2 pi) inter L^(1 +) (RR\/2 pi). $Let
  $
  (T_n f) (t) &:= integral.dash_0^(2 pi) f(x) cot^n ((x-t)/2) dd(x)
  $
  then
  $
  (T_(n + 1) f) (t)
  &:= -2/n dv(,t) (T_(n) f) (t) - (T_(n - 1) f) (t)
  $
  where
  $
  (T_1 f) (t) &:= integral.dash_0^(2 pi) f(x) cot ((x - t)/2) dd(x) \
  (T_0 f) (t)&:= integral_0^(2 pi) f(x) dd(x)
  $
] <hadamard-recurrence>
#theorem[
  $forall n in NN. forall f in C^(n, alpha) (RR\/2 pi) inter L^(1 +) (RR\/2 pi).$
  $
  dv(,t) integral.dash_0^(2 pi) f(x) cot^n ((x - t)/2) dd(x) = integral.dash_0^(2 pi) f'(x) cot^n ((x - t)/2) dd(x)
  $
] <derivative-of-fp>
#proof[
$
  dv(,t) integral.dash_0^(2 pi) f(x) cot^n ((x - t)/2) dd(x)
  &= dv(,t) integral.dash_0^(2 pi) f(x + t) cot^n (x/2) dd(x) \
  &= dv(,t) lim_(epsilon -> 0) F(t, epsilon) \
  &=_? lim_(epsilon -> 0) dv(,t) F(t, epsilon) \
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
  I_(m,0) &= 0, &quad I_(m,1) &= 2 pi i sgn(m) quad (m != 0)\
  I_(0,0) &= 2 pi, &quad I_(0,1) &= 0
  $
]
#proof[
  - $I_(m, 0), m in ZZ$: @fourier-integral.
  - $I_(0, 1) = 0$: Due to the asymmetry of integrand $cot t/2$.
  - $I_(m, 1), m in ZZ$: Follows #cite(<kress_linear_2014>, supplement: [Lemma 8.23.]).
    - $cot t/2 = (cos t/2)/(sin t/2) = i (e^((i t)/2) + e^(- (i t)/2))/(e^((i t)/2) - e^(- (i t)/2)) = i (e^(i t) + 1)/(e^(i t) - 1)$
    - $therefore forall m in NN. (e^(i m t) - 1) cot (t/2) = (e^(i t) - 1) (sum_(j = 0)^(m - 1) e^(i j t)) cot (t/2) = i (e^(i t) + 1) sum_(j = 0)^(m - 1) e^(i j t)$.
    - $therefore forall m in NN. I_(m,1) - I_(0, 1) = 2 pi i because $Integrating both hands and @fourier-integral
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
    = (2 pi) sum_(k, l in ZZ without {0}, k + l + m = 0) 1/(abs(k) abs(l)) = (2 pi) sum_(k in ZZ without {0}, l in ZZ, k + l + m = 0) 1/(abs(k) abs(l)) = (2 pi) sum_(k in ZZ without {0, -m}) 1/(abs(k) abs(m + k))
    $
    - $forall m in NN. sum_(k in ZZ without {0, -m}) 1/(abs(k) abs(m + k)) = sum_(k = 1)^(infinity) 1/(k (m + k)) + sum_(k = -m + 1)^(-1) 1/(abs(k) abs(m + k)) + sum_(k = -infinity)^(- m + 1) 1/(abs(k) abs(m + k)) = sum_(k = 1)^(infinity) 1/(k (m + k)) + sum_(k = 1)^(m - 1) 1/(k (m - k)) + sum_(k = 1)^(infinity) 1/(k (m + k)) = 2/m sum_(k = 1)^(infinity) (1/k - 1/(m + k)) + 1/m sum_(k = 1)^(m - 1) (1/k + 1/(m - k)) = 2/m (sum_(k = 1)^m 1/k + sum_(k = 1)^(m - 1) 1/k) = 2/m (2 H_m - 1/m)$
]
== Subspace $U_N$

#lemma[
  $forall N' in NN. t_j := (2 pi j)/N'. forall m in ZZ. forall ts in [0, 2 pi).$
  $
    (2 pi)/N' sum_(j=0)^(N'-1) e^(i m (t_j - ts)) = cases(2 pi &(m equiv 0 mod N'), 0 &("otherwise"))
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
    integral_0^(2 pi) f(t) g(t) dd(t) = (2 pi)/N' sum_(j=0)^(N'-1) f(t_j)
  $
] <dft-trapezoidal>
#proof[
  Since, $f(t) g(t) in U_(2 N - 1)$ @trapezoidal-rule can be applied with $N' = 2 N - 1$ points.
]

== Quadratures

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

#theorem[Generalized Garrick--Wittich quadrature for $U_N$][
  $forall n in NN_0. forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) f(t) cot^n (t/2) dd(t) = sum_(j=0)^(N'-1) f(t_j + ts) P_j^(N',n)
  $
  $
  P_j^(N',n) := 1/N' sum_(abs(m) < N) I_(m,n) e^(- i m (t_j + ts))
  $
]
#theorem[Generalized Kussmaul--Martensen (Kress) quadrature for $U_N$][
  $forall n in NN_0. forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'. forall ts in [0, 2 pi).$
  $
    integral.dash_0^(2 pi) f(t) log(4 sin^2 (t/2)) cot^n (t/2) dd(t) = sum_(j=0)^(N'-1) f(t_j + ts) Q_j^(N',n)
  $
  $
  Q_j^(N',n) := 1/N' sum_(abs(m) < N) J_(m,n) e^(- i m (t_j + ts))
  $
]

#bibliography("quadrature.bib")
