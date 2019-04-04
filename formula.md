$$
\tilde r_{ui}=b_u+b_i+(n_u^+)^{-\alpha}\sum_{j\in (R_u^+-\{i\})} p_jq_i^T\\
L_{rmse}(R|\Theta)=\frac{1}{2}\sum_{i\in R'}||\tilde r_{ui}-r_{ui}||^2+\frac{\beta}{2}(||P'||^2+||Q'||^2)+\frac{\gamma}{2}(||b'_u||^2+||b'_i||^2)\\
R'=Z\or R_u^+, Z=sample(R_u^-)\\
Q'=\{Q_{ik}|i \in R',k=1...K\}\\
P'=\{P_{jk}|j\in R_u^+,k=1...K\}\\
L_{armse}(R|\Theta)=L_{rmse}(R|\Theta)+\lambda L_{rmse}(R|\Theta+\Delta)\\
(\Delta_P)_{jk}=\epsilon \frac{(\Gamma_P)_{jk}}{(\sum_k (\Gamma_P)^2_{jk})^{0.5}}\\
(\Gamma  _P)_{jk}=\frac{\part L_{armse}(R|\Theta)}{\part P_{jk}},j=1...N,k=1...K\\
note:\\l2\_reg\ of\ adv\ item\ is\ \lambda\beta\ or\ \beta\ or\ 0?
$$

$$
L_{auc}(R|\Theta)=\frac{1}{2}\sum_{i\in R_u^+\and j\in Z}||\tilde r_{ui}-\tilde r_{uj}-1||^2+\frac{\beta}{2}(||P'||^2+||Q'||^2)+\frac{\gamma}{2}(||b'_u||^2+||b'_i||^2)\\
Q' = \{Q_{ik}|i\in (R_u^+\or Z),k=1...K\}\\
P' = \{P_{jk}|j\in R_u^+,k=1...K\}\\
L_{aauc}(R|\Theta)=L_{auc}(R|\Theta)+\lambda L_{auc}(R|\Theta+\Delta)\\
(\Delta_P)_{jk}=\epsilon \frac{(\Gamma_P)_{jk}}{(\sum_k (\Gamma_P)_{jk}^2)^{0.5}}\\
(\Gamma_P)_{jk}=\frac{\part L_{armse}(R|\Theta)}{\part P_{jk}},j=1...N,k=1...K\\
$$

$$
\frac{\part{f(x+y)}}{\part x}=\frac{\part{f(x+y)}}{\part y}
$$

