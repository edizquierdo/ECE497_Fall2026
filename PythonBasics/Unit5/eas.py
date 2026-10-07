import numpy as np
import matplotlib.pyplot as plt

class Generational():

    def __init__(self, fitnessFunction, genesize, generations, **kwargs):
        self.fitnessFunction = fitnessFunction
        self.genesize = genesize
        self.generations = generations
        self.popsize = kwargs['popsize']
        self.recombProb = kwargs['recombProb']
        self.mutatStd = kwargs['mutatStd']
        self.elite = int(kwargs['eliteprop']*self.popsize)
        self.pop = np.random.rand(self.popsize,genesize)*2 - 1
        self.fitness = np.zeros(self.popsize)
        self.rank = np.zeros(self.popsize,dtype=int)
        self.avgHistory = np.zeros(generations)
        self.bestHistory = np.zeros(generations)
        self.gen = 0

    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()
        return self.bestHistory

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        return avgfit, bestfit, bestind

    def save(self,filename):
        af,bf,bi = self.fitStats()
        np.savez(filename, avghist=self.avgHistory, besthist=self.bestHistory, bestind=bi)

    def run(self):

        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])

        # Evolutionary loop
        for g in range(self.generations):

            # Report statistics every generation
            self.gen = g
            self.fitStats()

            # Rank individuals by fitness
            tempfitness = self.fitness.copy()
            for i in range(self.popsize):
                self.rank[i]=int(np.argmax(tempfitness))
                tempfitness[self.rank[i]]=-np.inf

            # Start new generation
            new_pop = np.zeros((self.popsize,self.genesize))
            new_fitness = np.zeros(self.popsize)

            # Fill out the elite first
            for i in range(self.elite):
                new_pop[i] = self.pop[self.rank[i]]
                new_fitness[i] = self.fitness[self.rank[i]]

            # Fill out remainder of the population through reproduction of most fit parents
            for i in range(self.elite,self.popsize):
                # Pick parents based on rank probability
                a = self.rank[int(np.random.triangular(0, 0, self.popsize))]
                b = self.rank[int(np.random.triangular(0, 0, self.popsize))]
                while (a==b):           # Make sure they are two different individuals
                    b = self.rank[int(np.random.triangular(0, 0, self.popsize))]

                # Recombine parents to produce child
                for k in range(self.genesize):
                    if np.random.random() < self.recombProb:
                        new_pop[i][k] = self.pop[a][k]
                    else:
                        new_pop[i][k] = self.pop[b][k]

                # Mutate child and make sure they stay within bounds
                new_pop[i] += np.random.normal(0.0,self.mutatStd,size=self.genesize)
                new_pop[i] = np.clip(new_pop[i],-1,1)

                # Recalculate their fitness
                new_fitness[i] = self.fitnessFunction(new_pop[i])

            # Finally replace old population with the new one
            self.pop = new_pop.copy()
            self.fitness = new_fitness.copy()

class Microbial():

    def __init__(self, fitnessFunction, genesize, generations, **kwargs):
        self.fitnessFunction = fitnessFunction
        self.genesize = genesize
        self.generations = generations
        self.popsize = kwargs['popsize']
        self.recombProb = kwargs['recombProb']
        self.mutatStd = kwargs['mutatStd']
        self.demeSize = int(kwargs['demeSize']/2)
        self.tournaments = generations*self.popsize
        self.pop = np.random.rand(self.popsize,genesize)*2 - 1
        self.fitness = np.zeros(self.popsize)
        self.avgHistory = np.zeros(generations)
        self.bestHistory = np.zeros(generations)
        self.gen = 0

    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()
        return self.bestHistory

    def fitStats(self):
        self.bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        #print(self.gen,": ",avgfit," ",bestfit)
        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        return avgfit, bestfit, self.bestind

    def run(self):
        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])
        # Evolutionary loop
        for g in range(self.generations):
            self.gen = g
            # Report statistics every generation
            self.fitStats()
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0,self.popsize)
                b = np.random.randint(a-self.demeSize,a+self.demeSize+1)%self.popsize   ### Restrict to demes
                while (a==b):   # Make sure they are two different individuals
                    b = np.random.randint(a-self.demeSize,a+self.demeSize+1)%self.popsize   ### Restrict to demes
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner 
                r = np.random.random(self.genesize)
                newind = np.array([self.pop[winner][k] if r[k] >= self.recombProb else self.pop[loser][k] for k in range(self.genesize)])
                self.pop[loser] = newind
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.random.normal(0.0,self.mutatStd,size=self.genesize)
                self.pop[loser] = np.clip(self.pop[loser],-1,1)
                # Step 5: Update fitness
                self.fitness[loser] = self.fitnessFunction(self.pop[loser])

class HillClimber():

    def __init__(self, fitnessFunction, genesize, generations, **kwargs):
        # Save parameters as attributes in the class
        self.fitnessFunction = fitnessFunction
        self.genesize = genesize
        self.generations = generations
        self.mutatStd = kwargs['mutatStd']
        # Initialize climber at random
        self.ind = np.random.rand(genesize) * 2 - 1
        # 1. Calculate individual's fitness score
        self.fitness = self.fitnessFunction(self.ind)
        # Keep track of the history of the best fitness found so far
        self.bestHistory = np.zeros(generations)
        self.bestHistory[0] = self.fitness

    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()
        return self.bestHistory

    def fitStats(self):
        # Single individual, so "avg" and "best" are the same
        return self.fitness, self.fitness, self.ind

    def run(self):
        # 2. Start the loop for the number of generations
        for g in range(1,self.generations):
            #print(g,self.fitness)
            # 3. Create an offspring of the parent through a mutation
            newindividual = self.ind + np.random.normal(0.0,self.mutatStd,size=self.genesize)
            newindividual = np.clip(newindividual, -1, 1)
            # 4. Evaluate the fitness of this new individual 
            newfitness = self.fitnessFunction(newindividual)
            # 5. Keep the individual only if it is equal or better than parent
            #    Otherwise, discard offspring
            if newfitness >= self.fitness:
                self.ind = newindividual
                self.fitness = newfitness
            # 6. Keep track of fitness over time
            self.bestHistory[g] = self.fitness

class ParallelHillClimber():

    def __init__(self, fitnessFunction, genesize, generations, **kwargs):
        # Save parameters as attributes in the class
        self.genesize = genesize
        self.generations = generations
        self.mutatStd = kwargs['mutatStd']
        self.popsize = kwargs['popsize']
        # fitnessFunction evaluates one genotype at a time; apply it across the population
        self.fitnessFunction = lambda pop: np.array([fitnessFunction(ind) for ind in pop])
        # Initialize climber at random
        self.pop = np.random.rand(self.popsize,genesize) * 2 - 1
        # 1. Calculate everyones fitness score
        self.fitness = self.fitnessFunction(self.pop)
        # Keep track of the history of the best fitness in the population
        self.bestHistory = np.zeros(generations)
        self.bestHistory[0] = np.max(self.fitness)

    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()
        return self.bestHistory

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        return avgfit, bestfit, bestind

    def run(self):
        # 2. Start the loop for the number of generations
        for g in range(1,self.generations):
            # 3. Create an offspring of the parent through a mutation
            newpop = self.pop + np.random.normal(0.0,self.mutatStd,size=(self.popsize,self.genesize))
            newpop = np.clip(newpop, -1, 1)
            # 4. Evaluate the fitness of this new individual 
            newfitness = self.fitnessFunction(newpop)
            # 5. Keep the individual only if it is equal or better than parent
            #    Otherwise, discard offspring
            for i in range(self.popsize):
                if newfitness[i] >= self.fitness[i]:
                    self.pop[i] = newpop[i]
                    self.fitness[i] = newfitness[i]
            # 6. Keep track of fitness over time
            self.bestHistory[g] = np.max(self.fitness)

