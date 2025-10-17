load('data_PCA_E7.RData')

x.pca<-prcomp(X)

# some analysis for PCA
print(x.pca)
plot(x.pca,type='l')
print(summary(x.pca))

