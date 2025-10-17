load('data_NN_E7.RData')

library(nnet)

# for MLP neural network the data need to be scaled first
minx=apply(X,2,min)
maxx=apply(X,2,max)

miny=min(y)
maxy=max(y)

X_S=scale(X,minx,maxx-minx)
y_s=scale(y,miny,maxy-miny)

# train the NN model
model_nn <- nnet(X_S, y_s, size=20,
                 maxit=300, decay=0.03, linout=TRUE, reltol=1.e-6, MaxNWts=100000)

# produce the predictions
y_s.predict <- predict(model_nn, X_S)

print(cor(y_s.predict,y_s))

x.pca<-prcomp(X,center=TRUE,scale=TRUE)