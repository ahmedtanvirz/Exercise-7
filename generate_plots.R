# Load the data
load('data_NN_E7.RData')

library(nnet)

# Generate plots for different models to understand what might be "scattered"

# 1. Full Neural Network
minx <- apply(X, 2, min)
maxx <- apply(X, 2, max)
miny <- min(y)
maxy <- max(y)

X_S <- scale(X, minx, maxx - minx)
y_s <- scale(y, miny, maxy - miny)

model_nn <- nnet(X_S, y_s, size=20,
                 maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)

y_s.predict <- predict(model_nn, X_S)

# 2. PCANN with N=1 (low number of PCs)
x.pca <- prcomp(X[,c(1,2,4,5,6,7,8)], center=TRUE, scale=TRUE)
pred_pca <- predict(x.pca)

N <- 1
x_data <- data.frame(X[,c(3,9,10)], pred_pca[,1:N])
model_pcann_1 <- nnet(x_data, y_s, size=20,
                      maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)
y_spca.predict_1 <- predict(model_pcann_1, x_data)

# 3. PCANN with N=3 (moderate number of PCs)
N <- 3
x_data <- data.frame(X[,c(3,9,10)], pred_pca[,1:N])
model_pcann_3 <- nnet(x_data, y_s, size=20,
                      maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)
y_spca.predict_3 <- predict(model_pcann_3, x_data)

# 4. PCANN with N=7 (high number of PCs)
N <- 7
x_data <- data.frame(X[,c(3,9,10)], pred_pca[,1:N])
model_pcann_7 <- nnet(x_data, y_s, size=20,
                      maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)
y_spca.predict_7 <- predict(model_pcann_7, x_data)

# Create plots
png("comparison_plots.png", width=1200, height=800)
par(mfrow=c(2,3))

# Full NN
plot(y_s, y_s.predict, main="Full Neural Network", xlab="Actual", ylab="Predicted")
abline(0,1, col="red")
cor_full <- cor(y_s, y_s.predict)
text(0.1, 0.9, paste("Corr:", round(cor_full, 3)), cex=1.2)

# PCANN N=1
plot(y_s, y_spca.predict_1, main="PCANN N=1 (Low PCs)", xlab="Actual", ylab="Predicted")
abline(0,1, col="red")
cor_1 <- cor(y_s, y_spca.predict_1)
text(0.1, 0.9, paste("Corr:", round(cor_1, 3)), cex=1.2)

# PCANN N=3
plot(y_s, y_spca.predict_3, main="PCANN N=3 (Moderate PCs)", xlab="Actual", ylab="Predicted")
abline(0,1, col="red")
cor_3 <- cor(y_s, y_spca.predict_3)
text(0.1, 0.9, paste("Corr:", round(cor_3, 3)), cex=1.2)

# PCANN N=7
plot(y_s, y_spca.predict_7, main="PCANN N=7 (High PCs)", xlab="Actual", ylab="Predicted")
abline(0,1, col="red")
cor_7 <- cor(y_s, y_spca.predict_7)
text(0.1, 0.9, paste("Corr:", round(cor_7, 3)), cex=1.2)

# Residuals plot for N=1 (most scattered)
residuals_1 <- y_s - y_spca.predict_1
plot(y_spca.predict_1, residuals_1, main="Residuals: PCANN N=1", xlab="Predicted", ylab="Residuals")
abline(h=0, col="red")

# Residuals plot for N=3 (less scattered)
residuals_3 <- y_s - y_spca.predict_3
plot(y_spca.predict_3, residuals_3, main="Residuals: PCANN N=3", xlab="Predicted", ylab="Residuals")
abline(h=0, col="red")

dev.off()

cat("=== PLOT ANALYSIS ===\n")
cat("Full NN correlation:", round(cor_full, 3), "\n")
cat("PCANN N=1 correlation:", round(cor_1, 3), "\n")
cat("PCANN N=3 correlation:", round(cor_3, 3), "\n")
cat("PCANN N=7 correlation:", round(cor_7, 3), "\n")

cat("\n=== SCATTER ANALYSIS ===\n")
cat("N=1 shows the most scattered plot (lowest correlation)\n")
cat("This is because only 1 principal component captures limited variance\n")
cat("Higher N values show less scatter (higher correlations)\n")