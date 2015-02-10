
	Hepatitis: a normal hierarchical
			model with measurement error

This example is taken from Spiegelhalter et al (1996) (chapter in Markov Chain Monte Carlo in Practice)   and concerns 106 children whose post-vaccination anti Hb titre was measured 2 or 3 times.  Both measurements and times have been transformed to a log scale.  One covariate y0 = log titre at baseline, is available.

The model is essentially a random effects linear growth curve

	Yij ~  Normal(ai + bi (tij -tbar), t)

	ai  ~  Normal(ac, ta)

	bi  ~  Normal(bc, tb)

where  t represents the precision (1/variance) of a normal distribution. We note the absence of a parameter representing correlation between ai and bi unlike in Gelfand et al 1990. However, see the Birats example in Volume 2 which does explicitly model the covariance between ai   and bi.  

ac , ta , bc , tb , t are given independent ``noninformative'' priors. 

Graphical model for hep example:



BUGS language for hep example:
		model
		{
			for( i in 1 : N ) {
				for( j in 1 : T ) {
					Y[i , j] ~ dnorm(mu[i , j],tau)
					mu[i , j] <- alpha[i] + beta[i] * (t[i,j] - 6.5) + 
									gamma * (y0[i] - mean(y0[]))
				}
				alpha[i] ~ dnorm(alpha0,tau.alpha)
				beta[i] ~ dnorm(beta0,tau.beta)
			}
			tau        ~ dgamma(0.001,0.001)
			sigma   <- 1 / sqrt(tau)
			alpha0    ~ dnorm(0.0,1.0E-6)	   
			tau.alpha ~ dgamma(0.001,0.001)
			beta0     ~ dnorm(0.0,1.0E-6)
			tau.beta ~ dgamma(0.001,0.001)
			gamma    ~ dnorm(0.0,1.0E-6)
		}


Note the use of a very flat but conjugate prior for the population effects: a locally uniform prior could also have been used.

Data ( click to open )

Inits for chain 1 		Inits for chain 2	( click to open )



Results

		mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
	alpha0	6.139	0.151	0.001416	5.844	6.138	6.433	2001	20000
	beta0	-1.051	0.1364	0.007766	-1.295	-1.059	-0.7525	2001	20000
	gamma	0.6734	0.08496	0.002025	0.5063	0.6738	0.8384	2001	20000
	sigma	1.002	0.05478	8.903E-4	0.9006	1.001	1.116	2001	20000


With measurement error

 

		model
		{
			tau.alpha ~ dgamma(0.001,0.001)
			alpha0 ~ dnorm( 0.0,1.0E-6)
			beta0 ~ dnorm( 0.0,1.0E-6)
			tau.beta ~ dgamma(0.001,0.001)
			for( i in 1 : N ) {
				alpha[i] ~ dnorm(alpha0,tau.alpha)
				beta[i] ~ dnorm(beta0,tau.beta)
				y0[i] ~ dnorm(mu0[i],tau)
				mu0[i] ~ dnorm(theta,psi)
			}
			for( j in 1 : T ) {
				for( i in 1 : N ) {
					Y[i , j] ~ dnorm(mu[i , j],tau)
					mu[i , j] <- alpha[i] + beta[i] * (t[i , j] -  6.5) + 
						gamma * (mu0[i] - mean(y0[]))
				}
			}
			tau ~ dgamma(0.001,0.001)
			sigma <- 1 / sqrt(tau)
			gamma ~ dnorm( 0.0,1.0E-6)
			theta ~ dnorm( 0.0,1.0E-6)
			psi ~ dgamma(0.001,0.001)
		}



Data ( click to open )

Inits for chain 1 	Inits for chain 2	( click to open )


Results

		mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
	alpha0	6.144	0.1594	0.003372	5.832	6.143	6.466	2001	20000
	beta0	-1.077	0.1291	0.007081	-1.333	-1.075	-0.8378	2001	20000
	gamma	1.056	0.1797	0.008859	0.7588	1.036	1.481	2001	20000
	sigma	1.021	0.06052	0.002232	0.9121	1.017	1.15	2001	20000

