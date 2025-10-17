load('data_NN_E7.RData')

library(nnet)

# in this task only y need scaling, because, PCA produces components in the right scale 
miny=min(y)
maxy=max(y)
y_s=scale(y,miny,maxy-miny)

# PCA can be performed only for continuous variables
x.pca<-prcomp(X[,c(1,2,4,5,6,7,8)],center=TRUE,scale=TRUE)
pred_pca<-predict(x.pca)

# N is the number of PCs selected for the next model (max 7 in this example)
N=1

# Here, also the 0/1 variables will be added back to the data
x_data<-data.frame(X[,c(3,9,10)],pred_pca[,1:N])

model_pcann <- nnet(x_data, y_s, size=20,
                 maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)

y_spca.predict <- predict(model_pcann, x_data)

print(cor(y_spca.predict,y_s))

