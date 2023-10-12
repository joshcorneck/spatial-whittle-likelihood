library(spatstat)
library(reshape2)

# FULL SET OF PARAMETER VALUES
lam <- 0.01
param_set <- matrix(c(0.6 * lambda, 2, 0.3 * lambda, 6), ncol = 2, byrow = TRUE)

# DATA FRAME TO STORE ESTIMATES
param_ests <- data.frame(
    n = numeric(0),          
    Method = numeric(0), 
    Param_Set = character(0), 
    rho = numeric(0),        
    K = numeric(0),          
    sigma = numeric(0)       
  )

# ROW STORAGE
row_counter <- 0

# LOAD IN THE SPP DATA
for (n in c(25, 50, 100, 200, 400, 800)){

  cat(paste("Number of points: ", n, "\n"))

  elln <- (n / lam)^(1 / 2)
  for (i in 1:nrow(param_set)){

    cat(paste("Parameter set: ", i, "\n"))

    # PARAMETERS FOR THIS RUN
    rho <- param_set[i, 1]
    K <- lam / rho
    sigma <- param_set[i, 2]
    
    # FITTING METHOD
    for (j in c(0,1)){

      cat(paste("Fitting Method: ", j, "\n"))
      
      if (j == 0){
        # RUN 1000 ITERATIONS FOR CHOICE OF n AND params
        for (k in 1:1000){
          row_counter <- row_counter + 1

          if (k %% 100 == 0){
              cat(paste("Iteration:", k, " of 1000.", "\n"))
          }
          

          # SIMULATE THE PROCESS
          thom <- rThomasHom(rho, K, sigma, square(elln))

          # FIT USING MINIMUM CONTRAST
          fitMC <- kppm(thom ~ 1, "Thomas")
          params <- parameters(fitMC)
          param_ests[row_counter,] <- c(
            n, j, i-1,
            params$kappa,
            params$mu,
            params$scale
            )
      }
      }
      else {
          # RUN 1000 ITERATIONS FOR CHOICE OF n AND params
          for (k in 1:1000){
            row_counter <- row_counter + 1
              
            if (k %% 100 == 0){
                cat(paste("Iteration:", k, " of 1000.", "\n"))
            }

            # SIMULATE THE PROCESS
            thom <- rThomasHom(rho, K, sigma, square(elln))

            # FIT USING PALM
            fitPalm <- kppm(thom ~ 1, "Thomas", method = 'palm')
            params <- parameters(fitPalm)
            param_ests[row_counter,] <- c(
              n, j, i-1,
              params$kappa,
              params$mu,
              params$scale
              )
      }
      }

    } 

  }
}

write.csv(param_ests, '/fitting_methods/param_ests.csv', row.names=FALSE)