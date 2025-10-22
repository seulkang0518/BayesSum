import numpy as np

# Two huge, nearly-equal terms with opposite signs
a32   = np.float32(1e30)                # big but representable in float32
logf  = np.array([np.log(a32), np.log(a32)], dtype=np.float32)  # both same
w     = np.array([1.0, -(1.0 - 1e-8)], dtype=np.float32)        # nearly cancel

# Naive: sum then log
s_naive = np.sum(w * np.exp(logf))      # ≈ tiny but cancellation wrecks precision
print("naive s =", s_naive)             # often 0.0 or wrong sign
print("naive log(s) =", np.log(s_naive))  # -> -inf or nan

# Stable: split pos/neg and combine in log-space
m = np.max(logf)
v = logf - m
w_pos = np.maximum(w, 0.0); w_neg = np.maximum(-w, 0.0)
A = np.log(np.sum(w_pos * np.exp(v)))   # log(sum positive part)
B = np.log(np.sum(w_neg * np.exp(v)))   # log(sum negative part)
d = A - B
log_abs_s = max(A,B) + np.log1p(-np.exp(-abs(d)))  # log|exp(A)-exp(B)|
log_sum   = log_abs_s + m
sign      = 1.0 if d >= 0 else -1.0
print("stable log-sum =", log_sum, " sign=", sign)