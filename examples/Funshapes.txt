	Fun shapes: general constraints

These toy examples illustrate how to implement general inequality constraints. They generate points uniformly over a restricted region of the square with corners at (1, 1) and (-1, -1). A dummy bernoulli variable is introduced and its corresponding proportion set to the step function of the required constraint. The only useful example is the parallelagram where x + y is constrained to be less than or equal to one, this idea can be used to model proportions for binary data instead of say logistic regression.

Circle

	model
	{
		x ~ dunif(-1, 1)
		y ~ dunif(-1, 1)
		O <- 0
		O ~ dbern(constraint)
		constraint <- step(x * x + y * y - 1)
	}

Inits ( click to open )




Square minus circle

	model
	{
		x ~ dunif(-1, 1)
		y ~ dunif(-1, 1)
		O <- 1
		O ~ dbern(constraint)
		constraint <- step(x * x + y * y - 1)
	}

Inits ( click to open )




Ring

	model
	{
		x ~ dunif(-1, 1)
		y ~ dunif(-1, 1)
		O1 <- 0
		O1 ~ dbern(constraint1)
		constraint1 <- step(x * x + y * y - 1)	
		O2 <- 1
		O2 ~ dbern(constraint2)
		constraint2 <- step( x * x + y * y - 0.25)
	}

Inits ( click to open )




Hollow square

	model
	{
		x ~ dunif(-1, 1)
		y ~ dunif(-1, 1)
		O <- 0
		O ~ dbern(constraint)
		constraint <- (step(0.5 - abs(x)) * step(0.50 - abs(y)))
	}

Inits ( click to open )





Parallelagram

	model
	{
		x ~ dunif(0, 1)
		y ~ dunif(-1, 1.0)
		O1 <- 1
		O1 ~ dbern(constraint1)
		constraint1 <- step(x + y)
		O2 <- 0
		O2 ~ dbern(constraint2)	
		constraint2 <- step(x + y - 1)
	}

Inits ( click to open )



