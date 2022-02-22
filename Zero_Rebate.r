library(sde)
library(writexl)
library(vctrs)
library(stringr)
library(MASS)
library(installr)
library(dplyr)
library(rsample)
library(keras)
library(tensorflow)
#use_condaenv("r-reticulate", required = TRUE)

#Barrier Option_closed formula: 
Ba_call_bsc <- function(S, B, K, r, sigma,t=0, T){
  
  d1  = (log(S/K)+(r+0.5*sigma^2)*(T-t))/(sigma*sqrt(T-t))
  d2  = (log(B^2/(S*K))+(r+0.5*sigma^2)*(T-t))/(sigma*sqrt(T-t))
  d11 = (log(S/B)+(r+0.5*sigma^2)*(T-t))/(sigma*sqrt(T-t))
  d21 = (log(B^2/(S*B))+(r+0.5*sigma^2)*(T-t))/(sigma*sqrt(T-t))
  d3  = (log(B/S)+(r+0.5*sigma^2)*(T-t))/(sigma*sqrt(T-t))
  dp1 = d1-sigma*sqrt(T-t)
  dp2 = d2-sigma*sqrt(T-t)
  dp11 = d11-sigma*sqrt(T-t)
  dp21 = d21-sigma*sqrt(T-t)
  lam = (2*r + sigma^2)/(2*sigma^2)
  if(K>=B){
    res <- S*pnorm(d1)-K*exp(-r*(T-t))*pnorm(dp1)-(S*(B/S)^(2*lam)*pnorm(d2)-
                                                     K*exp(-r*(T-t))*(B/S)^(2*lam-2)*pnorm(dp2))
    
  }
  else{
    res <- S*pnorm(d11)-K*exp(-r*(T-t))*pnorm(dp11)-(S*(B/S)^(2*lam)*pnorm(d21)-
                                                       K*exp(-r*(T-t))*(B/S)^(2*lam-2)*pnorm(dp21))
  }
  return(res)
}


#Generate Dataset for the Network training
sssize<-GBM(x=150,r = 0.04,sigma = 0.5,T = 1, N=1000000)
Datab <- NULL
Datab$Stock <- sssize
Datab$Strike <- sssize*runif(length(sssize),min = 0.4,max=1)
Datab$Barrier <- sssize*runif(length(sssize),min = 0.4,max=1)
Datab$Time <- runif(n=length(sssize), min=0.5, max=1.5)
Datab$sigma <- runif(n=length(sssize), min = 0.1, max = 0.5)
#Data$Rebate <- runif(n=length(ssize), min = 0.01, max = 0.05)
Datab$r <-runif(n=length(sssize),min = 0.01, max=0.05)
Datab$EBSS <-  Ba_call_bsc(S = Datab$Stock, B= Datab$Barrier, K = Datab$Strike, 
                           t = 0,  r = Datab$r, T = Datab$Time, 
                           sigma = as.numeric(Datab$sigma));

FDatab <- data.frame(Datab$Stock, Datab$Strike, Datab$Barrier, 
                     Datab$Time, Datab$sigma, Datab$r, Datab$EBSS)
#FDatab <- write_xlsx(list(mysheet = FDatab), 'FDatab.xlsx')
names(FDatab) <- c("Stock", "Strike", "Barrier", "Time",
                   "sigma", "r", "EBSS")

View(FDatab)
#Split dataset
splite <- initial_split(FDatab, prop = 3/4, strata = 'EBSS')
traine <- training(splite)
teste  <- testing(splite)

# Create & standardize feature sets
# training features
x_traine <- traine %>% dplyr::select(-EBSS)
meanne    <- colMeans(x_traine)
stdde     <- apply(x_traine, 2, sd)
x_traine <- scale(x_traine, center = meanne, scale = stdde)

# testing features
x_teste <- teste %>% dplyr::select(-EBSS)
x_teste <- scale(x_teste, center = meanne, scale = stdde)
# Create & transform response sets
y_traine <- log(traine[['EBSS']])
y_teste  <- log(teste[['EBSS']])

#Create and Define Model

#MODEL A
model_A <- keras_model_sequential() %>%
  layer_dense(units =  200,activation = "relu", 
              input_shape = ncol(x_traine)) %>%
  layer_dense(units = 150,activation = "relu") %>%
  layer_dense(units = 50,activation = "relu") %>%
  layer_dense(units = 1)  %>%
  
  #backpropagation----compile
  compile(
    optimizer = "adam",
    loss = "mse",
    metrics = c("mae")
  )

#Training the model
learn_A <- model_A %>% fit(
  x = x_traine,
  y = y_traine,
  epochs = 45,
  batch_size = 256,
  validation_split = .2,
  verbose = TRUE
)
#####
plot(learn_A)


#Evaluate the model
model_A %>% evaluate(x_teste, y_teste, batch_size = 256)
model_A %>% evaluate(x_traine, y_traine, batch_size = 256)

#MODEL B
model_B <- keras_model_sequential() %>%
  layer_dense(units =  200,activation = "relu", input_shape = ncol(x_traine),
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 150,activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50,activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)  %>%
  
  compile(
    optimizer = "rmsprop", #optimizer_adam(lr = 0.01) 
    loss = "mse",
    metrics = c("mae")
  )
learn_B <- model_B %>% fit(
  x = x_traine,
  y = y_traine,
  epochs = 45,
  batch_size = 256,
  validation_split = .2,
  verbose = TRUE,
  callbacks = list(
    #callback_early_stopping(patience = 10),
    callback_reduce_lr_on_plateau(patience = 5))
)

plot(learn_B)


#Evaluate the model
model_B %>% evaluate(x_teste, y_teste, batch_size = 256)
model_B %>% evaluate(x_traine, y_traine, batch_size = 256)

#MODEL C
model_C <- keras_model_sequential() %>% 
  layer_dense(units = 200, input_shape = ncol(x_traine)) %>% 
  layer_activation_leaky_relu() %>% 
  #layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 150) %>%
  layer_activation_leaky_relu() %>% 
  #layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50) %>%
  layer_activation_leaky_relu() %>% 
  #layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1) %>%
  
  #backpropagation----compile
  compile(
    optimizer = "adam",
    loss = "mse",
    metrics = c("mae")
  )

#Training the model
learn_C <- model_C %>% fit(
  x = x_traine,
  y = y_traine,
  epochs = 45,
  batch_size = 256,
  validation_split = .2,
  verbose = TRUE
)
#####
plot(learn_C)

#Evaluate the model
model_C %>% evaluate(x_teste, y_teste, batch_size = 256)
model_C %>% evaluate(x_traine, y_traine, batch_size = 256)

#MODEL D
model_D <- keras_model_sequential() %>% 
  layer_dense(units = 200, input_shape = ncol(x_traine)) %>% 
  layer_activation_leaky_relu() %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 150) %>%
  layer_activation_leaky_relu() %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50) %>%
  layer_activation_leaky_relu() %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1) %>%
  
  #backpropagation----compile
  compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )

#Training the model
learn_D <- model_D %>% fit(
  x = x_traine,
  y = y_traine,
  epochs = 45,
  batch_size = 256,
  validation_split = .2,
  verbose = TRUE,
)
#####
plot(learn_D)

#Evaluate the model
model_D %>% evaluate(x_teste, y_teste, batch_size = 256)
model_D %>% evaluate(x_traine, y_traine, batch_size = 256)

###########

#Storing Prediction
Resulte <- NULL
Resulte$Te_value <- y_teste
Resulte$P_values_model_A<- model_A %>% predict(x_teste)
Resulte$P_values_model_B<- model_B %>% predict(x_teste)
Resulte$P_values_model_C<- model_C %>% predict(x_teste)
Resulte$P_values_model_D<- model_D %>% predict(x_teste)


####converting back to the original values
Resulte$Te_converted <- exp(Resulte$Te_value)
Resulte$P_converted_model_A <- exp(Resulte$P_values_model_A)
Resulte$P_converted_model_B <- exp(Resulte$P_values_model_B)
Resulte$P_converted_model_C <- exp(Resulte$P_values_model_C)
Resulte$P_converted_model_D <- exp(Resulte$P_values_model_D)

Resulte <- data.frame(Resulte$Te_value, Resulte$P_values_model_A, 
                      Resulte$P_values_model_B, Resulte$P_values_model_C, Resulte$P_values_model_D, 
                      Resulte$Te_converted, Resulte$P_converted_model_A,Resulte$P_converted_model_B,
                      Resulte$P_converted_model_C,Resulte$P_converted_model_D)

Resulte <- write_xlsx(list(mysheet = Resulte), 'Resulte.xlsx')

####Error Analysis
Resulte$Diff_A <- Resulte$Te_converted - Resulte$P_converted_model_A
Resulte$Diff_B <- Resulte$Te_converted - Resulte$P_converted_model_B
Resulte$Diff_C <- Resulte$Te_converted - Resulte$P_converted_model_C
Resulte$Diff_D <- Resulte$Te_converted - Resulte$P_converted_model_D


plot(Resulte$Diff_A, cex= 0.01, xlab='Test DataSet',ylab='Model A Diff Values',
     col="blue")
hist(Resulte$Diff_A)
plot(Resulte$Diff_B, cex= 0.01, xlab='Test DataSet',ylab='Model B Diff Values',
     col="blue")
hist(Resulte$Diff_B)
plot(Resulte$Diff_C, cex= 0.01, xlab='Test DataSet',ylab='Model C Diff Values',
     col="blue")
hist(Resulte$Diff_C)
plot(Resulte$Diff_D, cex= 0.01, xlab='Test DataSet',ylab='Model D Diff Values',
     col="blue")
hist(Resulte$Diff_D)


plot(x=Resulte$Te_converted, y=Resulte$P_converted_model_A, 
     cex= 0.1, xlab='Actual Values', ylab='Model A Values', col="blue")
plot(x=Resulte$Te_converted, y=Resulte$P_converted_model_B, 
     cex= 0.1, xlab='Actual Values', ylab='Model B Values', col="blue")
plot(x=Resulte$Te_converted, y=Resulte$P_converted_model_C, 
     cex= 0.1, xlab='Actual Values', ylab='Model C Values', col="blue")
plot(x=Resulte$Te_converted, y=Resulte$P_converted_model_D, 
     cex= 0.1, xlab='Actual Values', ylab='Model D Values', col="blue")
