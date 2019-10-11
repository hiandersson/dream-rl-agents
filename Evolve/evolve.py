
# externals
import hashlib
from random import randint
import time
import concurrent.futures
from tqdm import tqdm
import numpy as np
import sys
from operator import itemgetter

noFitness = 1000000

def sortHelper(genome):
	return -genome.fitness

class genomeResult:
	def __init__(self):
		self.genome = {}
		self.testScore = -1
		self.testCheckpoint = None

class genomeInPopulation:
	def __init__(self, evolverConfig, genomeUID, geneCreationClass):
		self.genome = {}
		self.fitness = noFitness
		self.evolverConfig = evolverConfig
		self.uid = genomeUID
		self.evaluated = False
		self.crossValidated = False
		self.geneCreationClass = geneCreationClass
		self.hash = 0
		self.result = {}

	def updateHash(self):
		genh = ""
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			genh += str(self.genome[key])
		self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

	def resetAndRandomizeAllGenes(self):
		self.genome = {}
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			ran = randint(0, len(value) - 1)
			self.genome[key] = value[randint(0, ran)]
		self.updateHash()

	def cloneGenesFrom(self, fromGenome):
		self.genome = {}
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			self.genome[key] = fromGenome.genome[key]

	def crossOverOneGene(self, mom, dad):
		# clone from mom
		self.cloneGenesFrom(mom)
		# get one gene from dad
		walk = 0
		geneIndexRandom = randint(0, len(self.evolverConfig['allPossibleGenesSimple']) - 1)
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if walk == geneIndexRandom:
				self.genome[key] = dad.genome[key]
			walk += 1
		self.updateHash()

	def crossOverTwoGenes(self, mom, dad):
		# clone from mom
		self.cloneGenesFrom(mom)
		# get one gene from dad
		walk = 0
		geneIndexRandom1 = randint(0, len(self.evolverConfig['allPossibleGenesSimple']) - 1)
		geneIndexRandom2 = randint(0, len(self.evolverConfig['allPossibleGenesSimple']) - 1)
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if walk == geneIndexRandom1:
				self.genome[key] = dad.genome[key]
			if walk == geneIndexRandom2:
				self.genome[key] = dad.genome[key]
			walk += 1
		self.updateHash()

	def mutateOneGene(self, verbose = 0):
		allowedIndexes = list()
		indexCount = 0
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if len(value) > 1:
				allowedIndexes.append(indexCount)
			indexCount = indexCount +1

		geneIndexRandom = allowedIndexes[randint(0, len(allowedIndexes) - 1)]

		walk = 0
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if walk == geneIndexRandom:
				ran = randint(0, len(value) - 1)
				self.genome[key] = value[ran]
			walk += 1
		self.updateHash()

	def mutateTwoGenes(self):
		geneIndexRandom = randint(0, len(self.evolverConfig['allPossibleGenesSimple']) - 1)
		walk = 0
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if walk == geneIndexRandom:
				ran = randint(0, len(value) - 1)
				self.genome[key] = value[ran]
				ran = randint(0, len(value) - 1)
				self.genome[key] = value[ran]
			walk += 1
		self.updateHash()

	def mutateOneGeneClose(self):
		geneIndexRandom = randint(0, len(self.evolverConfig['allPossibleGenesSimple']) - 1)
		geneRandomDirection = randint(0, 1)
		walk = 0
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			if walk == geneIndexRandom:
				currentIndex = value.index(self.genome[key])
				if currentIndex == 0 and geneRandomDirection == 0:
					geneRandomDirection = 1
				elif currentIndex == len(value) - 1 and geneRandomDirection == 1:
					geneRandomDirection = 0

				if geneRandomDirection == 0:
					currentIndex -= 1
					if currentIndex < 0:
						currentIndex = 0
				elif geneRandomDirection == 1:
					currentIndex += 1
					if currentIndex > len(value) - 1:
						currentIndex = len(value) - 1
				self.genome[key] = value[currentIndex]

			walk += 1
		self.updateHash()

	def printGenome(self):
		print("Genome id={:04} score={} genome={}".format(self.uid, int(self.fitness), self.genome))

	def getFitnessGenomeArgs(self, generation, tfDevice, pbar):

		genomeArgs = {
			'genome': self.genome,
			'pbar': pbar,
			'genomeUID': self.uid,
			'tfDevice': tfDevice,
			'episodes': self.evolverConfig['episodes'],
			'score_genome': self.evolverConfig['score_genome'],
		}

		return genomeArgs

def genomeThreadedSubmit(genomeArgs):

	score_genome = genomeArgs['score_genome']

	testResult = genomeResult()
	testResult.genome = genomeArgs['genome']
	testResult.genomeUID = genomeArgs['genomeUID']
	testResult.testScore, testResult.testCheckpoint = score_genome(genomeArgs['genome'], genomeArgs['episodes'], genomeArgs['pbar'])

	return testResult

class genomeMasterPopulation:
	def __init__(self):
		self.masterPopulationList = []

	def isDuplicate(self, genome):
		for i in range(0,len(self.masterPopulationList)):
			if (genome.hash == self.masterPopulationList[i].hash):
				return True

		return False

	def addGenome(self, genome):
		if self.isDuplicate(genome):
			print("addGenome() ERROR: hash clash - duplicate genome")
			return False

		self.masterPopulationList.append(genome)

		return True

class evovlePopulation:
	def __init__(self, evolverConfig):
		self.genomesList = []
		self.evolverConfig = evolverConfig
		self.genomeUID = 0
		self.genomeMasterPopulation = genomeMasterPopulation()
		self.genomeDoneCounter = 0

	def newGenomUID(self):
		self.genomeUID = self.genomeUID + 1
		return self.genomeUID

	def addToPopulation(self, newGenome):

		while self.genomeMasterPopulation.isDuplicate(newGenome):
			newGenome.mutateOneGene(verbose = 1)

		self.genomeMasterPopulation.addGenome(newGenome)
		self.genomesList.append(newGenome)

	def initialize(self):
		self.genomesList = []
		self.genomeDoneCounter = 0
		size = self.evolverConfig['populationSize']
		for i in range(0, size):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "Random")
			newGenome.resetAndRandomizeAllGenes()
			self.addToPopulation(newGenome)
		
		# scoring
		self.gene_scores = self.evolverConfig['allPossibleGenesSimple'].copy()
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			self.gene_scores[key] = {}
			for keyiter in value:
				self.gene_scores[key][keyiter] = 0

		# counter
		self.gene_counter = self.evolverConfig['allPossibleGenesSimple'].copy()
		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			self.gene_counter[key] = {}
			for keyiter in value:
				self.gene_counter[key][keyiter] = 0

	def printAllGenomes(self):
		for i in range(0, len(self.genomesList)):
			self.genomesList[i].printGenome()

	def evaluateAllGenomes(self, generation,  crossValidation):

		instance = self

		# GPU distribution
		maxWorkers = None
		rollingGPUDevices = None
		if self.evolverConfig['GPUDevices'] != None:
			rollingGPUDevices = self.evolverConfig['GPUDevices']
			rollingGPUDevicesIndex = 0
			maxWorkers = len(rollingGPUDevices)

		genCounterLeft = 0

		for i in range(0, len(instance.genomesList)):
			if self.genomesList[i].evaluated == False:
				genCounterLeft += 1

		position = 0
		
		# Genomes list
		while(1):
			
			genomeExecList = list()

			count = 0

			for i in range(0, len(instance.genomesList)):

				if self.genomesList[i].evaluated == False:
					self.genomesList[i].evaluated = True
					self.genomesList[i].pbar = tqdm(file=sys.stdout,total=self.evolverConfig['episodes'], desc = "Agent:{:03}".format(self.genomesList[i].uid), ncols = 128, position=position)

					if rollingGPUDevices == None:
						tfDevice = "/cpu:0"
					else:
						tfDevice = rollingGPUDevices[rollingGPUDevicesIndex % len(rollingGPUDevices)]
						rollingGPUDevicesIndex += 1

					genomeExecList.append(instance.genomesList[i].getFitnessGenomeArgs(generation, tfDevice, pbar=self.genomesList[i].pbar))
					position += 1

					if maxWorkers != None:
						count = count + 1
						if count > maxWorkers - 1:
							print("Wont submit more workers {} then max {}".format(count - 1, maxWorkers))
							break

			if maxWorkers != None and len(genomeExecList) == 0:
				print("Done. Nothing more to submit.")
				break

			# We can use a with statement to ensure threads are cleaned up promptly
			with concurrent.futures.ThreadPoolExecutor(maxWorkers) as executor:
				
				futureToGenomeExec = {executor.submit(genomeThreadedSubmit, genomeExec): genomeExec for genomeExec in genomeExecList}
				
				for future in concurrent.futures.as_completed(futureToGenomeExec):

					url = futureToGenomeExec[future]
					result = future.result()

					for i in range(0, len(instance.genomesList)):

						if result.genomeUID == instance.genomesList[i].uid:

							instance.genomesList[i].result = result

							if instance.genomesList[i].result.testScore == -1:
								instance.genomesList[i].fitness = noFitness
							else:
								instance.genomesList[i].fitness = instance.genomesList[i].result.testScore

			if maxWorkers == None:
				break
			
		for i in range(0, len(instance.genomesList)):
			if self.genomesList[i].pbar is not None:
				self.genomesList[i].pbar.close()
				self.genomesList[i].pbar = None

	def sortAndRetainGenomes(self):

		self.genomesList.sort(key=sortHelper)

		self.newGenomesList = []
		for i in range(self.evolverConfig['retainSize']):
			self.newGenomesList.append(self.genomesList[i])
		self.genomesList = self.newGenomesList

		return self.genomesList[0]

	def saveGeneScores(self):

		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			for i in range(0, len(self.genomesList)):
				self.gene_scores[key][self.genomesList[i].genome[key]] += self.genomesList[i].fitness ** 2
				self.gene_counter[key][self.genomesList[i].genome[key]] += 1

	def finalGeneScores(self):

		name_scores = list()

		for key,value in self.evolverConfig['allPossibleGenesSimple'].items():
			for i in range(0, len(value)):
				if self.gene_counter[key][value[i]] > 0:
					self.gene_scores[key][value[i]] /= self.gene_counter[key][value[i]]
					name_scores.append((key+" "+str(value[i]),self.gene_scores[key][value[i]]))

		name_scores.sort(key=itemgetter(1), reverse=True)

		for item in name_scores:
			print("{} : {}".format(item[0].ljust(20), int(item[1])))

	def cloneAndMutate(self):

		retainSize = self.evolverConfig['retainSize']
		retainedGenomes = self.genomesList

		# mutateOneGeneRandom
		numNewGenomes = self.evolverConfig['mutateOneGeneRandom']
		for i in range(numNewGenomes):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "mutateOneGeneRandom")
			newGenome.cloneGenesFrom(retainedGenomes[randint(0, retainSize - 1)])
			newGenome.mutateOneGene()
			self.addToPopulation(newGenome)

		# mutateTwoGenesRandom
		numNewGenomes = self.evolverConfig['mutateTwoGenesRandom']
		for i in range(numNewGenomes):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "mutateTwoGenesRandom")
			newGenome.cloneGenesFrom(retainedGenomes[randint(0, retainSize - 1)])
			newGenome.mutateTwoGenes()
			self.addToPopulation(newGenome)

		# crossoverOneGene
		numNewGenomes = self.evolverConfig['crossoverOneGene']
		for i in range(numNewGenomes):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "crossoverOneGene")
			newGenome.crossOverOneGene(retainedGenomes[randint(0, retainSize - 1)], retainedGenomes[randint(0, retainSize - 1)])
			self.addToPopulation(newGenome)

		# crossOverTwoGenes
		numNewGenomes = self.evolverConfig['crossOverTwoGenes']
		for i in range(numNewGenomes):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "crossOverTwoGenes")
			newGenome.crossOverTwoGenes(retainedGenomes[randint(0, retainSize - 1)], retainedGenomes[randint(0, retainSize - 1)])
			self.addToPopulation(newGenome)

		# mutateOneGeneClose
		numNewGenomes = self.evolverConfig['mutateOneGeneClose']
		for i in range(numNewGenomes):
			newGenome = genomeInPopulation(self.evolverConfig, self.newGenomUID(), "mutateOneGeneClose")
			newGenome.cloneGenesFrom(retainedGenomes[randint(0, retainSize - 1)])
			newGenome.mutateOneGeneClose()
			self.addToPopulation(newGenome)

class evolver:
	def __init__(self, evolverConfig):

		self.evovlePopulation = evovlePopulation(evolverConfig)
		self.evolverConfig = evolverConfig
		self.generationResult = []

	def start(self):

		self.evovlePopulation.initialize()

		generations = self.evolverConfig['generations']

		bestInGenerationHistory = []

		afterInitGenerations = generations - 1
		afterInitTotal = (self.evolverConfig['retainSize'] * afterInitGenerations) + (self.evolverConfig['mutateOneGeneRandom'] * afterInitGenerations) + (self.evolverConfig['mutateTwoGenesRandom'] * afterInitGenerations) + (self.evolverConfig['crossoverOneGene'] * afterInitGenerations) + (self.evolverConfig['crossOverTwoGenes'] * afterInitGenerations) + (self.evolverConfig['mutateOneGeneClose'] * afterInitGenerations)
		totalGenes = self.evolverConfig['populationSize'] + afterInitTotal

		# Evolution

		for i in range(generations):

			print("\n< ------- Generation {} ------- >\n".format(i))

			self.evovlePopulation.evaluateAllGenomes(i, False)

			bestGenome = self.evovlePopulation.sortAndRetainGenomes()
			self.evovlePopulation.saveGeneScores()
			
			print("\nWinning genome id {} - score {:.4} - gene inception '{}'".format(bestGenome.uid, bestGenome.fitness, bestGenome.geneCreationClass))
			print("Genome: {}".format(bestGenome.genome))

			bestInGenerationHistory.append(bestGenome.result.testScore)

			if i < generations - 1:
				self.evovlePopulation.cloneAndMutate()

		# Save best

		if self.evolverConfig['save_filepath'] is not None:
			print("Saving genome id {} - score {:.4} - to file {}".format(bestGenome.uid, bestGenome.fitness, self.evolverConfig['save_filepath']))
			save_checkpoint = self.evolverConfig['save_checkpoint']
			save_checkpoint(bestGenome.result.testCheckpoint, self.evolverConfig['save_filepath'])

		# Stats

		print("\n< ------- Statistics ------- >\n")

		print("Which genes where associated with the best scores?")
		print("Calculated as the aggreagate sum of agents best score per gene to the power of two, \ndivided by number of agents running that gene\n")

		self.evovlePopulation.finalGeneScores()

		bestGenome.result.generationHistory = bestInGenerationHistory

		return bestGenome.result.genome
