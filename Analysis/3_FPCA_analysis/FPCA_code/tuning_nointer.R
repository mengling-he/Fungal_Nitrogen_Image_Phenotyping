# Choose the optimal tuning parameter when there is no fixed effects
#
# This uses a simple GCV criterion for penalized least squares:
#   bhat = (X'X + lambda * Omega)^{-1} X'Y
#
# Notes:
# - Omega should be a KxK penalty matrix.
# - Prior versions mistakenly used adiag(Omega*lambda). The correct penalty is
#   (lambda * Omega).

tuning_nointer <- function(lower, upper, Omega, Xmat, Y_vec) {
  lam.list <- exp(seq(lower, upper, 1))
  gcv <- rep(NA_real_, length(lam.list))

  XtX <- t(Xmat) %*% Xmat
  XtY <- t(Xmat) %*% Y_vec
  nobs <- nrow(Xmat)

  for (ii in seq_along(lam.list)) {
    lam <- lam.list[ii]
    A <- solve(XtX + lam * Omega)
    Y_hat <- Xmat %*% (A %*% XtY)
    trH <- sum(diag(XtX %*% A))
    diag.mean <- trH / nobs
    gcv[ii] <- mean((Y_vec - Y_hat)^2) / (1 - diag.mean)^2
  }

  lam.list[which.min(gcv)]
}
