oneVsAll <- function(X, y, num_labels, lambda) {
  m=nrow(X)
  n=ncol(X)
  all_theta  = matrix(0,num_labels,n)
  initial_theta = matrix(0, n, 1);
  for (c in 1:num_labels){
   
    cur_y=y
    C_mat=matrix(c,m,1)
    new_y=(cur_y==C_mat)*1
    
    theta=gradientDescent(initial_theta,X,new_y,0.3,400)
   
    all_theta[c,] = t(theta);
  }
  return(all_theta)
}
sigmoid <- function(z) {
  m=nrow(X)
  g = matrix(0,m,1);
  g=1/(1+(exp(1)^-z));
  return(g)
  
}
lrCostFunction <- function(theta, X, y, lambda) {
  m=nrow(X)
  n=ncol(X)
  J = 0
  grad = matrix(0,n,1)
  z = X %*% theta
  g=sigmoid(z)
  thetaWithout=theta[2:m,]
  thetaForGrad=theta
  thetaForGrad[1]=0
  reg=lambda*(t(thetaWithout) %*% thetaWithout)/(2*m)
  J=((1/m)*((-t(y) %*% log(g)) - ((1.-t(y))*log(1-g))))+reg
  srt_error=(g %-% y)
  grad=(1/m) * (t(srt_error) %*% X ) + (t(lambda/m) * thetaForGrad)
  return(list(J,grad))
}
gradientDescent <- function(theta,X,y,alpha,max_iter){
  m = nrow(y)
  n =ncol(X)
  temp =  matrix(0,n,1)
  for (iter in 1:max_iter) {
    z = X %*% theta;
    predictions=sigmoid(z)
    srt_error=(predictions - y);
    for (i in 1:n) {
      temp[i,1]=theta[i,1] - ((alpha*(1/m)) * (t(X[,i]) %*% srt_error));
    }
    
    theta=temp;
    #J_history[iter] = costFunction(x, y, theta);
  }
  return(theta)
}

predictOneVsAll <- function(all_theta, X){
  z=X %*% t(all_theta)
  val=as.matrix(apply(z,1,max))
  pos=as.matrix(apply(z,1,which.max))
  pred= cbind(val,pos)
  #return(list(all_theta,pred))
  return(pred)
}

featureNormalize <- function(x){
  X_norm = x
  n=ncol(x)
  mu = matrix(0,1,n)
  sigma = matrix(0,1,n)
  for (rc in 1:n) {
    mu[1,rc]=mean(x[,rc])
    sigma[1,rc] = sd(x[,rc])
  }
  
  for (i in 1:n){
    X_norm[,i] = (1/sigma[i])*(x[,i]-mu[i]);
  }
  return(list(X_norm,mu,sigma))
}