data {
  int<lower=1> N;
  int<lower=1> n_species;

  array[N] int<lower=1, upper=n_species> species;
  array[N] int<lower=0> count;

  // 1 = Poisson
  // 2 = NegBin
  int<lower=1, upper=2> likelihood;

  real<lower=0> sigma_prior;
  real<lower=0> mu_prior_sd;
  real<lower=0> mu_prior_mean;
}

parameters {
  real mu;
  real<lower=0> sigma;

  vector[n_species] log_lambda;

  vector<lower=0>[n_species] phi;
}

model {
  mu ~ normal(mu_prior_mean, mu_prior_sd);

  sigma ~ exponential(1 / sigma_prior);

  log_lambda ~ normal(mu, sigma);

  phi ~ exponential(1);

  for (n in 1:N) {
  if (likelihood == 1) {
    count[n] ~ poisson_log(log_lambda[species[n]]);
  } else {
    count[n] ~ neg_binomial_2_log(log_lambda[species[n]], phi[species[n]]);
  }
  }
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  real lprior;

  real lprior_mu;
  real lprior_sigma;
  real lprior_phi;

  lprior_mu =
    normal_lpdf(mu | mu_prior_mean, mu_prior_sd);

  lprior_sigma =
    exponential_lpdf(
      sigma |
      1 / sigma_prior
    );

  lprior_phi =
    exponential_lpdf(phi | 1);

  if (likelihood == 1) {
    lprior = lprior_mu + lprior_sigma;
  } else {
    lprior = lprior_mu + lprior_sigma + lprior_phi;
  }
  
  for (n in 1:N) {

    if (likelihood == 1) {
      y_rep[n] =
        poisson_log_rng(
          log_lambda[species[n]]
        );

      log_lik[n] =
        poisson_log_lpmf(
          count[n] |
          log_lambda[species[n]]
        );

    } else {

      y_rep[n] =
        neg_binomial_2_log_rng(
          log_lambda[species[n]],
          phi[species[n]]
        );

      log_lik[n] =
        neg_binomial_2_log_lpmf(
          count[n] |
          log_lambda[species[n]],
          phi
        );
    }
  }  
}
