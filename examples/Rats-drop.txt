		Rats-drop: the rats example, illustrating the
						 effect of different dropout assumptions

This example is taken from  section 6 of Gelfand et al (1990), and concerns 30 young rats whose weights were measured weekly for five weeks. Part of the data is shown below, where Yij is the weight of the ith rat measured at age xj.   

		Weights Yij of rat i on day xj
		 xj = 8	15	22	29 	36 
	__________________________________		
	Rat 1	151	199	246	283	320
	Rat 2	145	199	249	293	354
	.......
	Rat 30	153	200	244	286	324 


A plot of the 30 growth curves suggests some evidence of downward curvature.

The model is essentially a random effects linear growth curve

							Yij ~  Normal(ai + bi(xj - xbar), tc)
							
							ai  ~  Normal(ac, ta)
							
							bi  ~  Normal(bc, tb)
							
where xbar = 22, and t represents the precision (1/variance) of a normal distribution. We note the absence of a parameter representing correlation between ai and bi unlike in Gelfand et al 1990. However, see the Birats example in Volume 2 which does explicitly model the covariance between ai   and bi. For now, we standardise the xj's around their mean to reduce dependence between ai and bi in their likelihood: in fact for the full balanced data, complete independence is achieved. (Note that, in general, prior independence does not force the posterior distributions to be independent).

ac , ta , bc , tb , tc are given independent ``noninformative'' priors.  Interest particularly focuses on the intercept at zero time (birth), denoted a0 = ac - bc xbar.  

Graphical model for rats example:


BUGS language for rats example:


	model
	{
		for( i in 1 : N ) {
			for( j in 1 : T ) {
				Y[i , j] ~ dnorm(mu[i , j],tau.c)
				mu[i , j] <- alpha[i] + beta[i] * (x[j] - xbar)
			}
			alpha[i] ~ dnorm(alpha.c,alpha.tau)
			beta[i] ~ dnorm(beta.c,beta.tau)
		}
		tau.c ~ dgamma(0.001,0.001)
		sigma <- 1 / sqrt(tau.c)
		alpha.c ~ dnorm(0.0,1.0E-6)	   
		alpha.tau ~ dgamma(0.001,0.001)
		beta.c ~ dnorm(0.0,1.0E-6)
		beta.tau ~ dgamma(0.001,0.001)
		alpha0 <- alpha.c - xbar * beta.c	
	}



Note the use of a very flat but conjugate  prior for the population effects: a locally uniform prior could also have been used.

Data ##=> click on one of the arrows to open the data 
list(x = c(8.0, 15.0, 22.0, 29.0, 36.0), xbar = 22, N = 30, T = 5,	
		Y = structure(
			.Data =   c(151, 199, 246, 283, 320,
							 145, 199, 249, 293, 354,
							 147, 214, 263, 312, 328,
							 155, 200, 237, 272, 297,
							 135, 188, 230, 280, 323,
							 159, 210, 252, 298, 331,
							 141, 189, 231, 275, 305,
							 159, 201, 248, 297, 338,
							 177, 236, 285, 350, 376,
							 134, 182, 220, 260, 296,
							 160, 208, 261, 313, 352,
							 143, 188, 220, 273, 314,
							 154, 200, 244, 289, 325,
							 171, 221, 270, 326, 358,
							 163, 216, 242, 281, 312,
							 160, 207, 248, 288, 324,
							 142, 187, 234, 280, 316,
							 156, 203, 243, 283, 317,
							 157, 212, 259, 307, 336,
							 152, 203, 246, 286, 321,
							 154, 205, 253, 298, 334,
							 139, 190, 225, 267, 302,
							 146, 191, 229, 272, 302,
							 157, 211, 250, 285, 323,
							 132, 185, 237, 286, 331,
							 160, 207, 257, 303, 345,
							 169, 216, 261, 295, 333,
							 157, 205, 248, 289, 316,
							 137, 180, 219, 258, 291,
							 153, 200, 244, 286, 324),
						.Dim = c(30,5)))##<=

(Note: the response data (Y) for the rats example can also be found in the file ratsy.odc in rectangular format. The covariate data (X) can be found in S-Plus format in file ratsx.odc. To load data from each of these files, focus the window containing the open data file before clicking on "Data" from the "Model" menu.)

Inits ##=> click on one of the arrows to open initial values 
list(  alpha.c = 150, beta.c = 10, 
		 tau.c = 1, alpha.tau = 1, beta.tau = 1)##<=
		
Gelfand et al 1990 also consider the problem of missing data, and delete the last observation of cases 6-10, the last two from 11-20, the last 3 from 21-25 and the last 4 from 26-30.  The appropriate data file is obtained by simply replacing data values by NA (see below). The model specification is unchanged, since the distinction between observed and unobserved quantities is made in the data file and not the model specification.

	##=> click on one of the arrows to open the data for the missing value analysis 
list(x = c(8.0, 15.0, 22.0, 29.0, 36.0), xbar = 22, N = 30, T = 5,	
		Y = structure(
			.Data =   c(151, 199, 246, 283, 320,
							 145, 199, 249, 293, 354,
							 147, 214, 263, 312, 328,
							 155, 200, 237, 272, 297,
							 135, 188, 230, 280, 323,
							 159, 210, 252, 298, NA,
							 141, 189, 231, 275, NA,
							 159, 201, 248, 297, NA,
							 177, 236, 285, 350, NA,
							 134, 182, 220, 260, NA,
							 160, 208, 261, NA, NA,
							 143, 188, 220, NA, NA,
							 154, 200, 244, NA, NA,
							 171, 221, 270, NA, NA,
							 163, 216, 242, NA, NA,
							 160, 207, 248, NA, NA,
							 142, 187, 234, NA, NA,
							 156, 203, 243, NA, NA,
							 157, 212, 259, NA, NA,
							 152, 203, 246, NA, NA,
							 154, 205, NA, NA, NA,
							 139, 190, NA, NA, NA,
							 146, 191, NA, NA, NA,
							 157, 211, NA, NA, NA,
							 132, 185, NA, NA, NA,
							 160, NA, NA, NA, NA,
							 169, NA, NA, NA, NA,
							 157, NA, NA, NA, NA,
							 137, NA, NA, NA, NA,
							 153, NA, NA, NA, NA),
						.Dim = c(30,5)),
						Missing = structure(
			.Data =   c(0, 0, 0, 0, 0,
							 0, 0, 0, 0, 0,
							 0, 0, 0, 0, 0,
							 0, 0, 0, 0, 0,
							 0, 0, 0, 0, 0,
							 0, 0, 0, 0, 1,
 					   	 0, 0, 0, 0, 1,
					    	 0, 0, 0, 0, 1,
						     0, 0, 0, 0, 1,
 			 	           0, 0, 0, 0, 1,
		            		 0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,
				             0, 0, 0, 1, 1,
						     0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,
		            		 0, 0, 0, 1, 1,		
		            		 0, 0, 0, 1, 1,
				             0, 0, 1, 1, 1,		 
				             0, 0, 1, 1, 1,				
				             0, 0, 1, 1, 1,				
				             0, 0, 1, 1, 1,				
				             0, 0, 1, 1, 1,				
				             0, 1, 1, 1, 1,				
				             0, 1, 1, 1, 1,				
				             0, 1, 1, 1, 1,				
				             0, 1, 1, 1, 1,
				             0, 1, 1, 1, 1
				),
						.Dim = c(30,5)))##<=
						
  Missing at random gives beta.c estimate of 6.54 (6.26 to 6.84)
						
						
						
		Extra data list for sensitivity analysis in informative dropout model
						list(k=0.02)
						

 Initial values for informative dropout model, probability of missing depending on Y[i,j] with a as intercept
Inits ##=> click on one of the arrows to open initial values for informative dropout model, single group 
list(  alpha.c = 250, beta.c =6,  tau.c = 1, alpha.tau = 1, beta.tau = 1,  a = 0)##<=
Second set of initial values
        ##=> click on one of the arrows to open initial values for informative dropout model, single group 
list(  alpha.c = 100, beta.c =4,  tau.c = .1, alpha.tau = .1, beta.tau = .1,  a = 1)##<=



With k=.02, get estimate of beta.c	 

node	 mean	 sd	 MC error	2.5%	median	97.5%	start	sample
	beta.c	6.273	0.15	0.009028	5.976	6.276	6.561	1001	3000
	


 
