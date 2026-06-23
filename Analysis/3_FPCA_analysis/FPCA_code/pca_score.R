# Principal component scores prediction (BLUP)
#
# Key fixes vs older versions:
# - Accept Y_vec explicitly (no global dependency)
# - Define n before allocating PC
# - Add numerical safeguards for sigma_e2

pca_score <- function(Xmat, bhat, Y_vec, T_list, K.est, v1.est, Phi1.est, sigma.e2.hat,
                      eps_sigma = 1e-8) {

  n <- length(T_list)
  K.pc <- K.est
  N <- nrow(Xmat)

  sigma2 <- max(as.numeric(sigma.e2.hat), eps_sigma)

  # Eigenvalues matrix Lambda (block-diagonal)
  G <- diag(rep(v1.est, n))

  # Eigenfunction matrix Phi (block-diagonal by subject)
  Z <- matrix(0, N, n * K.pc)
  start.temp <- 1
  for (i in seq_len(n)) {
    n.i <- length(T_list[[i]])
    Z[start.temp:(start.temp + n.i - 1), ((i - 1) * K.pc + 1):(i * K.pc)] <- Phi1.est[[i]]
    start.temp <- start.temp + n.i
  }

  # Sigma^{-1} using Woodbury identity: (sigma^2 I + Z G Z')^{-1}
  # S.inv = (1/sigma2)I - (1/sigma2^2) Z (G^{-1} + (1/sigma2) Z'Z)^{-1} Z'
  mid <- solve(solve(G) + (t(Z) %*% Z) / sigma2)
  S.inv <- (1 / sigma2) * diag(N) - (1 / sigma2^2) * Z %*% mid %*% t(Z)

  # BLUP
  e <- as.numeric(Y_vec - Xmat %*% bhat)
  xi <- G %*% t(Z) %*% (S.inv %*% e)
  PC <- t(matrix(xi, K.pc, n))

  colnames(PC) <- paste0('fpca', seq_len(K.pc))
  rownames(PC) <- names(T_list)
  PC
}
