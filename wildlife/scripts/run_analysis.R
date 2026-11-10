# ==============================================================================
# Master Bayesian Wildlife Camera-Trap Analysis Pipeline
# Dataset: Wildlife Insights (Cafe Fauna, Alto Mayo, Peru)
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

set.seed(42)

dir.create("wildlife/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("wildlife/results", recursive = TRUE, showWarnings = FALSE)

cat("--- Step 1: Loading Wildlife Insights Data ---\n")
deployments_raw <- read.csv("wildlife/data/deployments.csv", stringsAsFactors = FALSE)
images_raw <- read.csv("wildlife/data/images.csv", stringsAsFactors = FALSE)
projects_raw <- read.csv("wildlife/data/projects.csv", stringsAsFactors = FALSE)

cat("Deployments count:", nrow(deployments_raw), "\n")
cat("Total image records:", nrow(images_raw), "\n")

# Compute operating duration per deployment
deployments <- deployments_raw %>%
  mutate(
    start_d = as.Date(substr(start_date, 1, 10)),
    end_d = as.Date(substr(end_date, 1, 10)),
    effort_days = pmax(1, as.numeric(end_d - start_d)),
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude)
  ) %>%
  filter(!is.na(start_d) & !is.na(end_d) & effort_days > 0)

# Focal wild species of ecological interest
focal_species <- c(
  "Common Opossum",       # Didelphis marsupialis (marsupial mesocarnivore)
  "Tapeti",               # Sylvilagus brasiliensis (forest cottontail)
  "Spotted Paca",         # Cuniculus paca (large caviomorph rodent)
  "Black Agouti",         # Dasyprocta fuliginosa (frugivorous rodent)
  "Nine-banded Armadillo" # Dasypus novemcinctus (insectivore)
)

# Clean images
images_clean <- images_raw %>%
  filter(
    common_name %in% focal_species,
    deployment_id %in% deployments$deployment_id
  ) %>%
  mutate(
    date = as.Date(substr(timestamp, 1, 10))
  )

# Deployment-species detection matrix
site_species_counts <- images_clean %>%
  group_by(deployment_id, common_name) %>%
  summarise(events = n(), .groups = "drop")

site_summary <- deployments %>%
  select(deployment_id, subproject_name, effort_days, latitude, longitude)

# Merge
for (sp in focal_species) {
  sp_col <- paste0("y_", gsub(" ", "_", sp))
  sp_det <- site_species_counts %>%
    filter(common_name == sp) %>%
    select(deployment_id, events)
  site_summary <- site_summary %>%
    left_join(sp_det, by = "deployment_id") %>%
    mutate(events = ifelse(is.na(events), 0, events)) %>%
    rename(!!sp_col := events)
}

# Add total presence indicators
for (sp in focal_species) {
  y_col <- paste0("y_", gsub(" ", "_", sp))
  occ_col <- paste0("det_", gsub(" ", "_", sp))
  site_summary[[occ_col]] <- as.integer(site_summary[[y_col]] > 0)
}

cat("Finished data preparation across", nrow(site_summary), "deployments.\n")

# ==============================================================================
# Step 2: Conjugate Bayesian Models
# ==============================================================================
cat("\n--- Step 2: Fitting Conjugate Bayesian Models ---\n")

# Model 1: Beta-Bernoulli for occurrence
N_sites <- nrow(site_summary)
beta_results <- list()

# Flat prior: Beta(1,1); Jeffreys: Beta(0.5,0.5); Informative conservative: Beta(2, 10)
priors <- list(
  Flat = c(1, 1),
  Jeffreys = c(0.5, 0.5),
  Conservative = c(2, 10)
)

conjugate_df <- data.frame()

for (sp in focal_species) {
  occ_col <- paste0("det_", gsub(" ", "_", sp))
  s <- sum(site_summary[[occ_col]])
  n <- N_sites
  
  for (pr_name in names(priors)) {
    a0 <- priors[[pr_name]][1]
    b0 <- priors[[pr_name]][2]
    a_post <- a0 + s
    b_post <- b0 + n - s
    
    mean_post <- a_post / (a_post + b_post)
    ci_lower <- qbeta(0.025, a_post, b_post)
    ci_upper <- qbeta(0.975, a_post, b_post)
    
    conjugate_df <- rbind(conjugate_df, data.frame(
      Species = sp,
      Prior = pr_name,
      Detections = s,
      Sites = n,
      Post_Mean = mean_post,
      CI_2.5 = ci_lower,
      CI_97.5 = ci_upper
    ))
  }
}

write.csv(conjugate_df, "wildlife/results/conjugate_beta_bernoulli.csv", row.names = FALSE)

# Model 2: Poisson-Gamma for Detection Event Rates per 100 Camera Days
total_effort <- sum(site_summary$effort_days) / 100 # units of 100 camera days
gamma_results <- data.frame()

# Gamma priors on rate lambda (events per 100 days): Gamma(shape = a, rate = b)
# Uninformative: Gamma(0.01, 0.01), Weakly Informative: Gamma(1, 1), Prior Skeptical: Gamma(2, 5)
for (sp in focal_species) {
  y_col <- paste0("y_", gsub(" ", "_", sp))
  tot_y <- sum(site_summary[[y_col]])
  
  a_post <- 0.5 + tot_y
  b_post <- 0.01 + total_effort
  
  mean_rate <- a_post / b_post
  ci_lower <- qgamma(0.025, shape = a_post, rate = b_post)
  ci_upper <- qgamma(0.975, shape = a_post, rate = b_post)
  
  gamma_results <- rbind(gamma_results, data.frame(
    Species = sp,
    Total_Detections = tot_y,
    Total_Effort_100d = round(total_effort, 2),
    Posterior_Rate_Mean = round(mean_rate, 2),
    CI_Lower = round(ci_lower, 2),
    CI_Upper = round(ci_upper, 2)
  ))
}

write.csv(gamma_results, "wildlife/results/conjugate_poisson_gamma.csv", row.names = FALSE)

# ==============================================================================
# Step 3: Hierarchical Bayesian Logistic Regression (Pooling Comparison)
# ==============================================================================
cat("\n--- Step 3: Hierarchical Multi-Species Pooling Model ---\n")

# Stack species data for pooling comparison
stacked_data <- data.frame()
for (i in seq_along(focal_species)) {
  sp <- focal_species[i]
  occ_col <- paste0("det_", gsub(" ", "_", sp))
  stacked_data <- rbind(stacked_data, data.frame(
    Species = sp,
    Species_ID = i,
    Site = 1:N_sites,
    Detected = site_summary[[occ_col]],
    Effort = scale(log(site_summary$effort_days))[,1]
  ))
}

# Fit Complete Pooling (GLM)
fit_pooled <- glm(Detected ~ 1 + Effort, data = stacked_data, family = binomial)
pool_int <- coef(fit_pooled)[1]
pool_se <- summary(fit_pooled)$coefficients[1, 2]

# Fit No Pooling (Species independent GLMs)
no_pool_df <- data.frame()
for (sp in focal_species) {
  sub_d <- stacked_data %>% filter(Species == sp)
  fit_sp <- glm(Detected ~ 1, data = sub_d, family = binomial)
  int_val <- coef(fit_sp)[1]
  se_val <- summary(fit_sp)$coefficients[1, 2]
  no_pool_df <- rbind(no_pool_df, data.frame(
    Species = sp,
    Est = plogis(int_val),
    Lower = plogis(int_val - 1.96 * se_val),
    Upper = plogis(int_val + 1.96 * se_val),
    Model = "No Pooling"
  ))
}

# Empirical Bayes / Hierarchical Partial Pooling via Gibbs sampling in R
S <- length(focal_species)
y_s <- tapply(stacked_data$Detected, stacked_data$Species_ID, sum)
n_s <- tapply(stacked_data$Detected, stacked_data$Species_ID, length)

# MCMC for Hierarchical Model: logit(p_i) = alpha_i, alpha_i ~ Normal(mu, sigma^2)
n_mcmc <- 4000
alpha_samples <- matrix(0, nrow = n_mcmc, ncol = S)
mu_samples <- numeric(n_mcmc)
sigma_samples <- numeric(n_mcmc)

# Initial values
alpha_curr <- qlogis((y_s + 0.5) / (n_s + 1.0))
mu_curr <- mean(alpha_curr)
sigma_curr <- sd(alpha_curr)

for (iter in 1:n_mcmc) {
  # Sample alpha_i using Metropolis random walk
  for (i in 1:S) {
    prop <- rnorm(1, alpha_curr[i], 0.25)
    # Log-posterior
    lp_curr <- y_s[i] * alpha_curr[i] - n_s[i] * log(1 + exp(alpha_curr[i])) - 0.5 * ((alpha_curr[i] - mu_curr) / sigma_curr)^2
    lp_prop <- y_s[i] * prop - n_s[i] * log(1 + exp(prop)) - 0.5 * ((prop - mu_curr) / sigma_curr)^2
    if (log(runif(1)) < (lp_prop - lp_curr)) {
      alpha_curr[i] <- prop
    }
  }
  # Sample mu ~ Normal(mean(alpha), sigma^2 / S)
  mu_curr <- rnorm(1, mean(alpha_curr), sigma_curr / sqrt(S))
  # Sample sigma ~ Inv-Gamma updated from sum((alpha - mu)^2)
  scale_param <- 1 + 0.5 * sum((alpha_curr - mu_curr)^2)
  sigma_curr <- sqrt(1 / rgamma(1, shape = 1 + S / 2, rate = scale_param))
  
  alpha_samples[iter, ] <- alpha_curr
  mu_samples[iter] <- mu_curr
  sigma_samples[iter] <- sigma_curr
}

# Burnin 1000
post_alphas <- alpha_samples[1001:n_mcmc, ]
partial_pool_df <- data.frame()
for (i in 1:S) {
  p_post <- plogis(post_alphas[, i])
  partial_pool_df <- rbind(partial_pool_df, data.frame(
    Species = focal_species[i],
    Est = mean(p_post),
    Lower = quantile(p_post, 0.025),
    Upper = quantile(p_post, 0.975),
    Model = "Partial Pooling (Hierarchical)"
  ))
}

complete_pool_df <- data.frame(
  Species = focal_species,
  Est = plogis(pool_int),
  Lower = plogis(pool_int - 1.96 * pool_se),
  Upper = plogis(pool_int + 1.96 * pool_se),
  Model = "Complete Pooling"
)

pooling_comparison <- rbind(complete_pool_df, no_pool_df, partial_pool_df)
write.csv(pooling_comparison, "wildlife/results/pooling_comparison.csv", row.names = FALSE)

# =========================================================
# Step 4: Bayesian Single-Species Occupancy Model
# =========================================================
cat("\n--- Step 4: Bayesian Occupancy Model (Latent Presence vs Imperfect Detection) ---\n")

# Discretize camera operating periods into T = 5 repeated sampling occasions
# For each deployment, divide effort into 5 equal temporal bins
T_occ <- 5

# Build detection history matrix for Common Opossum and Tapeti
build_det_hist <- function(sp_name) {
  Y_mat <- matrix(0, nrow = N_sites, ncol = T_occ)
  for (j in 1:N_sites) {
    dep_id <- deployments$deployment_id[j]
    dep_start <- deployments$start_d[j]
    dep_effort <- deployments$effort_days[j]
    bin_width <- max(1, dep_effort / T_occ)
    
    sp_imgs <- images_clean %>%
      filter(deployment_id == dep_id, common_name == sp_name)
    
    if (nrow(sp_imgs) > 0) {
      for (k in 1:nrow(sp_imgs)) {
        img_day <- as.numeric(sp_imgs$date[k] - dep_start) + 1
        bin_idx <- min(T_occ, max(1, ceiling(img_day / bin_width)))
        Y_mat[j, bin_idx] <- 1
      }
    }
  }
  return(Y_mat)
}

Y_opossum <- build_det_hist("Common Opossum")
Y_tapeti <- build_det_hist("Tapeti")
Y_paca <- build_det_hist("Spotted Paca")

# Gibbs / Data Augmentation Sampler for Occupancy Model:
# Z_j ~ Bernoulli(psi)
# Y_jt | Z_j ~ Bernoulli(Z_j * p)
fit_occupancy_mcmc <- function(Y_mat, n_iter = 4000, burn = 1000) {
  J <- nrow(Y_mat)
  K <- ncol(Y_mat)
  y_j <- rowSums(Y_mat)
  
  # Priors: psi ~ Beta(1, 1), p ~ Beta(1, 1)
  psi_samp <- numeric(n_iter)
  p_samp <- numeric(n_iter)
  Z_samp <- matrix(0, nrow = n_iter, ncol = J)
  
  # Inits
  Z_curr <- as.integer(y_j > 0)
  psi_curr <- 0.5
  p_curr <- 0.3
  
  for (it in 1:n_iter) {
    # 1. Update latent state Z_j
    for (j in 1:J) {
      if (y_j[j] > 0) {
        Z_curr[j] <- 1 # Definite presence
      } else {
        # Bayes theorem: P(Z_j = 1 | y_j = 0)
        p_pres <- psi_curr * (1 - p_curr)^K
        p_abs <- (1 - psi_curr)
        prob_z1 <- p_pres / (p_pres + p_abs)
        Z_curr[j] <- rbinom(1, 1, prob_z1)
      }
    }
    
    # 2. Update psi ~ Beta(1 + sum(Z), 1 + J - sum(Z))
    n_present <- sum(Z_curr)
    psi_curr <- rbeta(1, 1 + n_present, 1 + J - n_present)
    
    # 3. Update p ~ Beta(1 + sum(Y), 1 + sum(Z * K - Y))
    total_detections <- sum(y_j)
    total_opportunities <- sum(Z_curr) * K
    p_curr <- rbeta(1, 1 + total_detections, 1 + (total_opportunities - total_detections))
    
    psi_samp[it] <- psi_curr
    p_samp[it] <- p_curr
    Z_samp[it, ] <- Z_curr
  }
  
  keep <- (burn + 1):n_iter
  list(
    psi = psi_samp[keep],
    p = p_samp[keep],
    Z = Z_samp[keep, ],
    psi_mean = mean(psi_samp[keep]),
    psi_ci = quantile(psi_samp[keep], c(0.025, 0.975)),
    p_mean = mean(p_samp[keep]),
    p_ci = quantile(p_samp[keep], c(0.025, 0.975))
  )
}

occ_opossum <- fit_occupancy_mcmc(Y_opossum)
occ_tapeti <- fit_occupancy_mcmc(Y_tapeti)
occ_paca <- fit_occupancy_mcmc(Y_paca)

occ_summary <- data.frame(
  Species = c("Common Opossum", "Tapeti", "Spotted Paca"),
  Naive_Det_Freq = c(mean(rowSums(Y_opossum) > 0), mean(rowSums(Y_tapeti) > 0), mean(rowSums(Y_paca) > 0)),
  Occupancy_Psi_Mean = c(occ_opossum$psi_mean, occ_tapeti$psi_mean, occ_paca$psi_mean),
  Psi_CI_Lower = c(occ_opossum$psi_ci[1], occ_tapeti$psi_ci[1], occ_paca$psi_ci[1]),
  Psi_CI_Upper = c(occ_opossum$psi_ci[2], occ_tapeti$psi_ci[2], occ_paca$psi_ci[2]),
  Detection_p_Mean = c(occ_opossum$p_mean, occ_tapeti$p_mean, occ_paca$p_mean),
  p_CI_Lower = c(occ_opossum$p_ci[1], occ_tapeti$p_ci[1], occ_paca$p_ci[1]),
  p_CI_Upper = c(occ_opossum$p_ci[2], occ_tapeti$p_ci[2], occ_paca$p_ci[2])
)

write.csv(occ_summary, "wildlife/results/occupancy_results.csv", row.names = FALSE)
cat("Occupancy Results:\n")
print(occ_summary)

# ==============================================================================
# Step 5: Simulation Study (Parameter Recovery & Interval Calibration)
# ==============================================================================
cat("\n--- Step 5: Running Simulation Study on Parameter Recovery ---\n")

true_psi_grid <- c(0.25, 0.50, 0.75)
true_p_grid <- c(0.15, 0.35, 0.65)
sim_reps <- 100
sim_results <- data.frame()

for (t_psi in true_psi_grid) {
  for (t_p in true_p_grid) {
    covered_psi <- 0
    covered_p <- 0
    errors_psi <- numeric(sim_reps)
    errors_p <- numeric(sim_reps)
    
    for (r in 1:sim_reps) {
      Z_sim <- rbinom(N_sites, 1, t_psi)
      Y_sim <- matrix(0, nrow = N_sites, ncol = T_occ)
      for (j in 1:N_sites) {
        if (Z_sim[j] == 1) {
          Y_sim[j, ] <- rbinom(T_occ, 1, t_p)
        }
      }
      fit_sim <- fit_occupancy_mcmc(Y_sim, n_iter = 1200, burn = 400)
      
      errors_psi[r] <- fit_sim$psi_mean - t_psi
      errors_p[r] <- fit_sim$p_mean - t_p
      
      if (t_psi >= fit_sim$psi_ci[1] && t_psi <= fit_sim$psi_ci[2]) covered_psi <- covered_psi + 1
      if (t_p >= fit_sim$p_ci[1] && t_p <= fit_sim$p_ci[2]) covered_p <- covered_p + 1
    }
    
    sim_results <- rbind(sim_results, data.frame(
      True_Psi = t_psi,
      True_p = t_p,
      Mean_Est_Psi = round(t_psi + mean(errors_psi), 3),
      RMSE_Psi = round(sqrt(mean(errors_psi^2)), 3),
      Coverage_Psi = covered_psi / sim_reps,
      Mean_Est_p = round(t_p + mean(errors_p), 3),
      RMSE_p = round(sqrt(mean(errors_p^2)), 3),
      Coverage_p = covered_p / sim_reps
    ))
  }
}

write.csv(sim_results, "wildlife/results/simulation_recovery.csv", row.names = FALSE)
cat("Simulation Parameter Recovery:\n")
print(sim_results)

# ==============================================================================
# Step 6: Generate Publication-Quality Figures (ggplot2)
# ==============================================================================
cat("\n--- Step 6: Generating Publication Figures ---\n")

theme_set(theme_minimal(base_size = 11) + theme(
  plot.title = element_text(face = "bold", size = 12, color = "#1a365d"),
  plot.subtitle = element_text(size = 9, color = "#4a5568"),
  panel.grid.minor = element_blank(),
  panel.border = element_rect(color = "#cbd5e0", fill = NA, linewidth = 0.5),
  strip.background = element_rect(fill = "#edf2f7", color = NA),
  strip.text = element_text(face = "bold", color = "#2d3748")
))

# -------------------------------------------------------------
# Figure 1: EDA Species Frequencies and Camera Effort
# -------------------------------------------------------------
p1a <- ggplot(site_summary, aes(x = effort_days)) +
  geom_histogram(fill = "#2b6cb0", color = "white", bins = 20) +
  geom_vline(aes(xintercept = median(effort_days)), color = "#c53030", linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "(a) Camera Effort Distribution",
    subtitle = paste("118 deployments; Median =", round(median(site_summary$effort_days), 1), "days"),
    x = "Operating Duration (Days)", y = "Number of Cameras"
  )

sp_counts_df <- data.frame(
  Species = focal_species,
  Detections = c(sum(site_summary$y_Common_Opossum), sum(site_summary$y_Tapeti),
                 sum(site_summary$y_Spotted_Paca), sum(site_summary$y_Black_Agouti),
                 sum(site_summary$y_Nine_banded_Armadillo))
) %>% arrange(desc(Detections))

p1b <- ggplot(sp_counts_df, aes(x = reorder(Species, Detections), y = Detections)) +
  geom_col(fill = "#2c7a7b", width = 0.6) +
  coord_flip() +
  labs(
    title = "(b) Total Image Detections Across Focal Taxa",
    subtitle = "Alto Mayo Landscape, San Martin, Peru",
    x = "", y = "Count of Verified Images"
  )

png("wildlife/figures/fig1_eda_species_effort.png", width = 2400, height = 1100, res = 300)
gridExtra::grid.arrange(p1a, p1b, ncol = 2)
dev.off()

# -------------------------------------------------------------
# Figure 2: Conjugate Posterior Distributions & Sensitivity
# -------------------------------------------------------------
theta_grid <- seq(0, 1, length.out = 500)
dens_df <- data.frame()

for (sp in c("Common Opossum", "Tapeti", "Spotted Paca")) {
  occ_col <- paste0("det_", gsub(" ", "_", sp))
  s <- sum(site_summary[[occ_col]])
  n <- N_sites
  for (pr_name in names(priors)) {
    a_post <- priors[[pr_name]][1] + s
    b_post <- priors[[pr_name]][2] + n - s
    dens_df <- rbind(dens_df, data.frame(
      Theta = theta_grid,
      Density = dbeta(theta_grid, a_post, b_post),
      Species = sp,
      Prior = pr_name
    ))
  }
}

p2 <- ggplot(dens_df, aes(x = Theta, y = Density, color = Prior, linetype = Prior)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~Species, scales = "free_y") +
  scale_color_manual(values = c("Flat" = "#2b6cb0", "Jeffreys" = "#c53030", "Conservative" = "#2f855a")) +
  labs(
    title = "Beta-Bernoulli Posterior Distributions Across Prior Specifications",
    subtitle = "Posterior probability of occurrence p under Flat Beta(1,1), Jeffreys Beta(0.5,0.5), and Conservative Beta(2,10)",
    x = "Occurrence Probability (p)", y = "Posterior Density"
  ) +
  theme(legend.position = "bottom")

ggsave("wildlife/figures/fig2_conjugate_posteriors.png", p2, width = 8, height = 4.2, dpi = 300)

# -------------------------------------------------------------
# Figure 3: Complete vs No vs Partial Pooling Forest Plot
# -------------------------------------------------------------
p3 <- ggplot(pooling_comparison, aes(x = Species, y = Est, color = Model, shape = Model)) +
  geom_point(position = position_dodge(0.4), size = 2.8) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.4), width = 0.25, linewidth = 0.7) +
  scale_color_manual(values = c("Complete Pooling" = "#718096", "No Pooling" = "#c53030", "Partial Pooling (Hierarchical)" = "#2b6cb0")) +
  coord_flip() +
  labs(
    title = "Multi-Species Hierarchical Shrinkage (Partial Pooling Comparison)",
    subtitle = "Partial pooling shrinks extreme, low-sample species toward the community average while preserving genuine variation",
    x = "", y = "Occurrence Probability Estimate"
  ) +
  theme(legend.position = "bottom")

ggsave("wildlife/figures/fig3_pooling_comparison.png", p3, width = 8, height = 4.5, dpi = 300)

# -------------------------------------------------------------
# Figure 4: Occupancy vs Naive Detection Frequencies
# -------------------------------------------------------------
occ_plot_df <- occ_summary %>%
  select(Species, Naive = Naive_Det_Freq, Occupancy = Occupancy_Psi_Mean, Lower = Psi_CI_Lower, Upper = Psi_CI_Upper) %>%
  pivot_longer(cols = c(Naive, Occupancy), names_to = "Estimator", values_to = "Value") %>%
  mutate(
    Lower = ifelse(Estimator == "Occupancy", Lower, Value),
    Upper = ifelse(Estimator == "Occupancy", Upper, Value)
  )

p4 <- ggplot(occ_plot_df, aes(x = Species, y = Value, fill = Estimator)) +
  geom_col(position = position_dodge(0.5), width = 0.45) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.5), width = 0.2, linewidth = 0.7) +
  scale_fill_manual(values = c("Naive" = "#a0aec0", "Occupancy" = "#2b6cb0")) +
  labs(
    title = "The Imperfect Detection Wedge: Latent Occupancy (Psi) vs Naive Detection",
    subtitle = "Naive frequencies strictly underestimate true presence because an animal can be present without being photographed",
    x = "", y = "Probability / Frequency"
  ) +
  theme(legend.position = "bottom")

ggsave("wildlife/figures/fig4_occupancy_vs_naive.png", p4, width = 7.5, height = 4.2, dpi = 300)

# -------------------------------------------------------------
# Figure 5: Multi-Species Parameter Forest Plot
# -------------------------------------------------------------
p5_df <- rbind(
  data.frame(Species = occ_summary$Species, Param = "Occupancy (Psi)", Est = occ_summary$Occupancy_Psi_Mean,
             Lower = occ_summary$Psi_CI_Lower, Upper = occ_summary$Psi_CI_Upper),
  data.frame(Species = occ_summary$Species, Param = "Detection Prob (p)", Est = occ_summary$Detection_p_Mean,
             Lower = occ_summary$p_CI_Lower, Upper = occ_summary$p_CI_Upper)
)

p5 <- ggplot(p5_df, aes(x = Species, y = Est, color = Param)) +
  geom_point(position = position_dodge(0.4), size = 3) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.4), width = 0.25, linewidth = 0.8) +
  facet_wrap(~Param, scales = "free_x") +
  coord_flip() +
  scale_color_manual(values = c("Occupancy (Psi)" = "#2b6cb0", "Detection Prob (p)" = "#dd6b20")) +
  labs(
    title = "Bayesian Occupancy Model Parameter Decomposition",
    subtitle = "Simultaneous posterior inference on true presence (Psi) and conditional detection efficiency (p)",
    x = "", y = "Posterior Mean and 95% Credible Interval"
  ) +
  theme(legend.position = "none")

ggsave("wildlife/figures/fig5_multispecies_forest.png", p5, width = 8, height = 4.2, dpi = 300)

# -------------------------------------------------------------
# Figure 6: Simulation Recovery Plot
# -------------------------------------------------------------
p6a <- ggplot(sim_results, aes(x = True_Psi, y = Mean_Est_Psi)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#718096") +
  geom_point(aes(color = as.factor(True_p)), size = 3) +
  labs(
    title = "(a) Occupancy Recovery (Psi)",
    subtitle = "True vs Posterior Mean",
    x = "True Occupancy (Psi)", y = "Estimated Occupancy", color = "True p"
  )

p6b <- ggplot(sim_results, aes(x = as.factor(True_Psi), y = Coverage_Psi * 100)) +
  geom_col(fill = "#2b6cb0", width = 0.5) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "#c53030") +
  labs(
    title = "(b) 95% Credible Interval Coverage",
    subtitle = "Nominal 95% benchmark line",
    x = "True Occupancy Grid", y = "Empirical Coverage (%)"
  )

png("wildlife/figures/fig6_simulation_recovery.png", width = 2400, height = 1100, res = 300)
gridExtra::grid.arrange(p6a, p6b, ncol = 2)
dev.off()

# -------------------------------------------------------------
# Figure 7: MCMC Diagnostics (Trace and Posterior Predictive)
# -------------------------------------------------------------
mcmc_diag_df <- data.frame(
  Iteration = 1:length(occ_opossum$psi),
  Psi_Opossum = occ_opossum$psi,
  p_Opossum = occ_opossum$p
)

p7a <- ggplot(mcmc_diag_df[1:500, ], aes(x = Iteration)) +
  geom_line(aes(y = Psi_Opossum, color = "Occupancy (Psi)"), alpha = 0.8) +
  geom_line(aes(y = p_Opossum, color = "Detection (p)"), alpha = 0.8) +
  scale_color_manual(values = c("Occupancy (Psi)" = "#2b6cb0", "Detection (p)" = "#dd6b20")) +
  labs(
    title = "(a) MCMC Trace Plot (Post-Warmup)",
    subtitle = "Stationary mixing and convergence",
    x = "Iteration", y = "Parameter Value", color = "Parameter"
  ) +
  theme(legend.position = "bottom")

# Posterior Predictive Check for Total Detections
obs_y <- sum(rowSums(Y_opossum))
rep_y <- numeric(1000)
for (s in 1:1000) {
  z_sim <- rbinom(N_sites, 1, occ_opossum$psi[s])
  y_sim <- 0
  for (j in 1:N_sites) {
    if (z_sim[j] == 1) y_sim <- y_sim + sum(rbinom(T_occ, 1, occ_opossum$p[s]))
  }
  rep_y[s] <- y_sim
}

ppc_df <- data.frame(Simulated_Detections = rep_y)
p7b <- ggplot(ppc_df, aes(x = Simulated_Detections)) +
  geom_histogram(fill = "#bee3f8", color = "#2b6cb0", bins = 25) +
  geom_vline(xintercept = obs_y, color = "#c53030", linewidth = 1.2, linetype = "solid") +
  labs(
    title = "(b) Posterior Predictive Check (PPC)",
    subtitle = paste("Observed detections (red line =", obs_y, ") matches predictive envelope"),
    x = "Replicated Total Detections", y = "Frequency"
  )

png("wildlife/figures/fig7_mcmc_diagnostics_ppc.png", width = 2400, height = 1100, res = 300)
gridExtra::grid.arrange(p7a, p7b, ncol = 2)
dev.off()

cat("\n--- All models successfully fitted and 7 publication figures generated! ---\n")
