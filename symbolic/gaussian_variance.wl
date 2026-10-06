(* Exact Gaussian variance; R is CHI, not chi-square. *)
ClearAll[d, n, t, r, v, chiTransform];
chiTransform[d_, t_] := Integrate[
  r Exp[-t r] PDF[ChiDistribution[d], r], {r, 0, Infinity},
  Assumptions -> d > 0 && t >= 0];
(* erfc MULTIPLIES Exp[t^2/2]; the same integration variable appears throughout. *)
v[d_, n_] := Integrate[
  chiTransform[d, t] (Exp[t^2/2] Erfc[t/Sqrt[2]])^(n - 1),
  {t, 0, Infinity}, Assumptions -> Element[{d,n}, Integers] && d >= 1 && n >= 1]/d;
(* Proven special cases; symbolic integrators may leave general d,n unevaluated. *)
expectedSpecialCases = {v[d,1] == 1/d, v[1,n] == 1/n,
  v[2,2] == Log[1+Sqrt[2]]/(2 Sqrt[2])};
(* No Mathematica execution is claimed by the Python verification suite. *)
