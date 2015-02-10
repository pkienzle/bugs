	Line: Linear Regression



	model
	{
		for( i in 1 : N ) {
			Y[i] ~ dnorm(mu[i],tau)
			mu[i] <- alpha + beta * (x[i] - xbar)
		}
		tau ~ dgamma(0.001, 0.001) sigma <- 1 / sqrt(tau)
		alpha ~ dnorm(0.0,1.0E-6)
		beta ~ dnorm(0.0,1.0E-6)	
	}




Data	( click to open )

Inits for chain 1	Inits for chain 2	( click to open )


Results 

A 1000 update burn in followed by a further 10000 updates gave the parameter estimates

		mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
	alpha	2.995	0.5436	0.005075	1.93	3.001	4.023	1001	10000
	beta	0.7947	0.3999	0.003365	0.04795	0.794	1.51	1001	10000
	sigma	1.003	0.7065	0.01486	0.4156	0.8229	2.694	1001	10000

