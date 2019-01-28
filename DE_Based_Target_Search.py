################################### Library Inclusion ############################################
import random, os, subprocess #,itertools
from sets import Set
import numpy as np
import math
import sys, errno
import re
import argparse
from time import time
import random
import string
import commands
import statistics
import hashlib
import itertools
from itertools import groupby
from itertools import cycle
import time
from collections import Counter
########################################### DBSCAN CLUSTERING ALGORITHM ############################################################
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
# Plot result
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

################################## Output Result Parameter ######################################
AlgoName='DE'
ResultFileName=AlgoName+'-FinalResult.csv'

################################## DE Parameter Setting ######################################

PopSize 		= None;			#Size of the population (i.e. Number of Chromosomes)
Iterations   		= None			#Number of Iterations
CR           		= None			#Crossover Rate
F            		= None			#Mutation Rate
Entropy                 = None			# Entropy Value
FunEval      		= 0;			#Function Evaluation
MaxFunEval   		= 1000000;		#Maximum allowable function evaluations
Runs         		= None			#Number of Runs
Deviation 		= 0			# Deviation in Entropy value
Pop          		= []			# Store the chromosome and its fitness
GlobalBestPop 		= []			# Global Best population
GlobalBestSector	= None			# Global Best sector
GlobalBestFitnessValue 	= 0 			#Global Best Fitnees value of Global Best Population
LB              	= []			#[0,0,0]
UB              	= []			#[14907,64,32]
Dim          		= None 			#Sector access Cylinder/Disk, Heads/Cylinder, Tracks/Head = 1 , Sector/Track. Eg: Cylinder/Disk 14907, 							Heads/Cylinder 64, Tracks/Head = 1, Sectors/Track = 32 or 14907/64/32
SectorSize      	= 512 	           	#Sector Size
SectorRead		= 512
StorageID 		= ''			#"/dev/sdb1"
SectorCount		= 0
MD5HashOfTargetFile 	= []			# MD5 Hash Value of each sector of file
SectorCollection	= []			# Collecting all important sectors of each run and iteration
Found			= []			# Final sector ranges of similar data
Max			= [0,0,0,0]
LBOfFitness		= None			#Lower bound of Fitness value
UBOfFitness		= None			#Upper bound of Fitness Value
TargetFileED		= []
TargetDataEDMean 	= None
TargetDataEDStDev 	= None
Start			= 32
DataExists		= []
ListOfAllDerivedSectors = []
NumberofGlobalNonRepeatedSectors = None
SectorEvaluated			 = 0
########################################### DE TIMING ##############################
DETime 		 	= None
SectorValues            = []

def TPTNFPFN():
	global MD5HashOfTargetFile
	global SectorEvaluated
	global Found
	global SectorCollection	
	global NumberofGlobalNonRepeatedSectors
	AllPositives   = len(MD5HashOfTargetFile)
	TruePositives  = len(Found)
	if AllPositives > TruePositives:
		FalseNegatives = abs(AllPositives - TruePositives)
		#TrueNegatives  = SectorCount - AllPositives
	elif AllPositives < TruePositives:
		Increase = int(math.ceil(TruePositives/float(AllPositives)))
		FalseNegatives = abs((Increase*AllPositives) - TruePositives)
	TrueNegatives  = SectorCount - SectorEvaluated
	FalsePositives = SectorEvaluated - TruePositives
	print "True Positive: ", TruePositives, "\t False Negative = ", FalseNegatives,"\t True Negative: ", TrueNegatives, "\t False Positive = ", FalsePositives
	print "Sensitivity: ", TruePositives/float(TruePositives+FalseNegatives)
	print "Specificity: ", TrueNegatives/float(TrueNegatives+FalsePositives)
	print "Fall Out: ", 1-(TrueNegatives/float(TrueNegatives+FalsePositives))
	print "Precision: ", TruePositives/float(TruePositives+FalsePositives)

class Individual(Exception): 			#Stores the Chromosome and its Fitness collectively
    def __init__(self,Chromosome,SectorNumber,Fitness):
        self.Chromosome   = Chromosome
	self.SectorNumber = SectorNumber
        self.Fitness      = Fitness

def main(file1):
        global Pop
	global DETime
	global SectorCollection
	global GlobalBestPop
	global GlobalBestSector
	global GlobalBestFitnessValue
	global NumberofGlobalNonRepeatedSectors
	BestCollection = []
	print "R = ", Runs,"\n IT = ", Iterations,"\n Population Size = ", PopSize ,"\n CR = ", CR, "\n Mut. = ", F,"\n Block Size = ", SectorRead  
	drive = file(StorageID,'rb')
	AllGlobalFindings = open(AlgoName+"-All_GlobalBest_Values.csv",'wa')
	StructureValues = None
	SectorDerived  = None
	FitnessDerived = None
	DETime = time.time()
	for R in range(1,Runs+1): # 0 to Runs or 1 to Runs+1 
		FunEval = 0 #Function Evaluation
		Pop=[]
		FileInitialPopulation=open(AlgoName+"Run-"+str(R)+"-Initial-Population.csv",'wa')
		GlobalBestFitnessValue=0 # Decides the reset of Fitness value at each Run of DE	
		PopGen(drive)
		StorePop(FileInitialPopulation)
		FileInitialPopulation.close()
		if GlobalBestFitnessValue == 0:
			StructureValues = Pop[0].Chromosome
			SectorDerived  = Pop[0].SectorNumber
			FitnessDerived = Pop[0].Fitness
			GlobalBestPop = Pop[0].Chromosome
			GlobalBestSector = Pop[0].SectorNumber
			GlobalBestFitnessValue = Pop[0].Fitness
		elif GlobalBestFitnessValue != 0:
			StructureValues = GlobalBestPop
			SectorDerived = GlobalBestSector
			FitnessDerived = GlobalBestFitnessValue
#		if GlobalBestFitnessValue != Pop[0].Fitness:
#        	    file2.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
#		    file1.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
#	    	    AllGlobalFindings.write(str(R) + "," +"0"+","+str(GlobalBestPop)+","+str(GlobalBestSector)+","+str(GlobalBestFitnessValue) + "\n")
		MemoriseGlobalBest();
		if FitnessDerived>=GlobalBestFitnessValue:
			SectorCollection.append(SectorDerived)
		else:
			SectorCollection.append(GlobalBestSector)
		file2=open(AlgoName+"-Result-Run-"+str(R)+".csv",'wa')
 		print "Run-->",R
		for IT in range(0,Iterations):
	        	if FunEval >= MaxFunEval:
			    break;
		        if IT%1==0:
	        	    #print R,"Run-->",IT,"IT-->",GlobalBestFitnessValue
	        	    file2.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
			    file1.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
			    if GlobalBestSector not in BestCollection:
			    	BestCollection.append([GlobalBestSector,GlobalBestFitnessValue])		    
			AllGlobalFindings.write(str(R) + "," +str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
			if GlobalBestSector not in SectorCollection:
				SectorCollection.append(GlobalBestSector)
			SortPop()
		        DEOperation(drive);	# Do the CROSSOVER and MUTATION of DE
		        MemoriseGlobalBest();	# MEMORISE GLOBAL BEST		# End of Iteration loop
			if GlobalBestSector not in SectorCollection:
				SectorCollection.append(GlobalBestSector)
				file2.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+","+str(GlobalBestFitnessValue) + "\n")
				file1.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+ "," +str(GlobalBestFitnessValue) + "\n")
		file2.write(str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+","+str(GlobalBestFitnessValue) + "\n")
		file2.close()
		AllGlobalFindings.write(str(R) + "," +str(IT) + "," +str(GlobalBestPop)+","+str(GlobalBestSector)+","+str(GlobalBestFitnessValue) + "\n")
	AllGlobalFindings.close()
	print time.time()- DETime," Seconds DE operation executed"
	if SectorCollection != []:
		writelisttofile('RelevantSectors',SectorCollection)
		SectorCollection.sort()
		writelisttofile('SortedOrderRelevantSectors',SectorCollection)
		fileforSectors=open(AlgoName+"-All_Considered_Sectors_Details_Sorted.csv",'wa')
		StoreSectors(fileforSectors,SectorCollection)
		fileforSectors.close()
		writelisttofile('EachandEverySector',ListOfAllDerivedSectors)
		print SectorCollection,'\n File searching started..................'
	if SectorCollection != []:
		#### DENSITY BASED CLUSTERING ####
		#dbscn(BestCollection)
		NumberofGlobalNonRepeatedSectors = len(BestCollection)
		regions=meanshift(BestCollection)
		FindinTime=time.time()
		find(drive,regions)#find(drive,SectorCollection)#SectorCollection[0],SectorCollection[len(SectorCollection)-1])
		print "Traversing Took ",time.time()- FindinTime," Seconds"
	TPTNFPFN() # Computing TP, TN, FP and FN
	drive.close()

###################### Pop Generation #####################
def PopGen(drive):
    global ListOfAllDerivedSectors
    global Pop
    Pop=[]
    for i in range(0,PopSize):
        Chromosome=[]
        for j in range(0,Dim):
            Chromosome.append(random.randint(LB[j],UB[j]))#int(random.uniform(LB[j],UB[j])))
	sectornumber = SecNum(Chromosome)
	Fitness = FitnessFunction(drive,Chromosome,sectornumber)
        Pop.append(Individual(Chromosome,sectornumber,Fitness)) 
	ListOfAllDerivedSectors.append(sectornumber)

def SecNum(child):
	sectornumber= long(0)
	global Max
	global SectorValues
	for i in range(0,len(child)): # Dim-1 or len(child)-1
		product = 1
		for j in range(i+1,len(UB)):
			product = product*UB[j]
		product = product*child[i]
	    	sectornumber = sectornumber + product
	if sectornumber <= SectorCount and sectornumber not in SectorValues:
		SectorValues.append(sectornumber)
		if sectornumber>Max[0]:
			Max[0]=sectornumber
			Max[1]=child[0]
			Max[2]=child[1]
			Max[3]=child[2]
		return sectornumber
	else:#aGAIN rANDOM nUMBER IS gENERATED
		secnum=[]    #Deriving new Sector Number
        	for j in range(0,Dim):
            		secnum.append(random.randint(LB[j],UB[j]))#int(random.uniform(LB[j],UB[j])))
		return(SecNum(secnum))


def FitnessFunction(drive,Chromosome,sectornumber):
    global GlobalBestPop
    global GlobalBestSector
    global GlobalBestFitnessValue
    global FunEval
    global DataFlag
    global SectorCount
    global DataExists
    FunEval+=1
    c=Chromosome
    data = None
    if long(sectornumber)<=long(SectorCount):
	drive.seek(Start)
	sectr=long(sectornumber)*long(SectorSize)
        drive.seek(sectr)
        data=drive.read(SectorRead)
	H = hsh(data)
	DataFlag=0
	fitnessvalue=fitvalue(data)
	if H in	MD5HashOfTargetFile and H != 'bf619eac0cdf3f68d496ea9344137e8b':# fitnessvalue != 0 or H!='bf619eac0cdf3f68d496ea9344137e8b':
		GlobalBestPop=Chromosome
		GlobalBestSector=sectornumber
		DataExists.append(sectornumber)
		print "Traces of Target Data Found at sector", sectornumber,"---->>",H
		print " Function evaluaton done for ", FunEval	
		print time.time()-DETime," Seconds after DE operation started"	
		fitnessvalue=UBOfFitness
		GlobalBestFitnessValue=fitnessvalue
        return fitnessvalue
    else:
	return 0

def fitvalue(data):
    datainhex = data.encode('hex')
    si = iter(datainhex)
    stList=map(''.join, itertools.izip(si, si))
    alphabet = list(Set(stList)) # list of symbols in the string
    # calculate the frequency of each symbol in the string
    DictfreqList={}
    freqList = []
    for symbol in alphabet:
        ctr = 0
        for sym in stList:
            if sym == symbol:
                ctr += 1
        freqList.append(float(ctr) / len(stList))
	DictfreqList.update({symbol:ctr})
    ent = 0.0
    for freq in freqList:
        ent = ent + freq * math.log(freq, 2)
    if not ent == 0.0:
    	ent = -ent
################## DATA DENSITY ###################
    for symbol in alphabet:
        ctr = 0
        for sym in stList:
            if sym != '00': #Apply Key concept if Dictionary is available
                ctr += 1
        dd=float(ctr)/float(len(stList))
################## Correlation Function ###############
    freqMean = statistics.mean(freqList)
    numerator=0
    denominator1=0
    denominator2=0
    for i in range(len(freqList)-1):
        numerator = numerator + ((freqList[i]-freqMean)*(freqList[i+1]-freqMean))
        denominator1 = denominator1 +(freqList[i]-freqMean)**2
        denominator2 = denominator2 +(freqList[i+1]-freqMean)**2
    try:
        cor=float(numerator)/float(math.sqrt(denominator1*denominator2))
    except ZeroDivisionError:
        cor=0
    try:
    	final=float(ent)/float(cor)#/float(encratio)#(float(ent))/(float(dd)*float(encratio))
    except ZeroDivisionError:
        if cor == 0:
	    final=float(ent)
    return abs(final)

def writelisttofile(name,data):
	FileToWrite=open(AlgoName+'-'+name+".csv",'wa')
	for i in data:
		FileToWrite.write(str(i)+'\n')
	FileToWrite.close()

def hsh(data):
    m=hashlib.md5(data).hexdigest()
    return m

def StorePop(file1):
	for i in range(0,len(Pop)):#PopSize):
		file1.write(str(i)+","+str(Pop[i].Chromosome)+ "," +str(Pop[i].SectorNumber)+ "," +str(Pop[i].Fitness) + "\n")

def StoreSectors(FileToWrite,Sectors):
	for i in Sectors:#PopSize):
		FileToWrite.write(str(i) + "\n")

def MemoriseGlobalBest():
    global GlobalBestFitnessValue
    global GlobalBestSector
    global GlobalBestPop
    for i in range(0,len(Pop)):
        if (Pop[i].Fitness == UBOfFitness):#(Pop[i].Fitness >= LBOfFitness and  Pop[i].Fitness <= UBOfFitness):
            GlobalBestFitnessValue = Pop[i].Fitness
	    GlobalBestSector = Pop[i].SectorNumber
            GlobalBestPop = Pop[i].Chromosome
	    break
        elif (Pop[i].Fitness != UBOfFitness) and (Pop[i].Fitness >= LBOfFitness and  Pop[i].Fitness <= UBOfFitness):# and Pop[i].Fitness>=GlobalBestFitnessValue:#(Pop[i].Fitness >= LBOfFitness and  Pop[i].Fitness <= UBOfFitness):
            GlobalBestFitnessValue = Pop[i].Fitness
	    GlobalBestSector = Pop[i].SectorNumber
            GlobalBestPop = Pop[i].Chromosome

def DEOperation(drive):
	global Pop
	for i in range(0,PopSize):		#For Every Population
		#Choosing three unique randoms
		r1=random.randint(0,len(Pop)-1)#int(random.uniform(0,len(Pop)))#PopSize)
		r2=random.randint(0,len(Pop)-1)#int(random.uniform(0,len(Pop)))#PopSize)
		r3=random.randint(0,len(Pop)-1)#int(random.uniform(0,len(Pop)))#PopSize)
		NewChild=[]	#Create a new child (trial vector) by manupulating each Dimension
		for j in range(0,Dim):	# Iterate for every Dimension
			if random.random() <= CR:		# On the basis of CR
				k = Pop[r1].Chromosome[j] + int(F * (Pop[r2].Chromosome[j] - Pop[r3].Chromosome[j]))
				if k < LB[j]:
					k = random.randint(LB[j],UB[j])#int(random.uniform(LB[j],UB[j])) #in case of crossing the Lb
				if k > UB[j]:
					k = random.randint(LB[j],UB[j])#int(random.uniform(LB[j],UB[j])) #in case of crossing the Ub
				NewChild.append(k)
			else:
				NewChild.append(Pop[random.randint(0,len(Pop)-1)].Chromosome[j])#Pop[i].Chromosome[j])
		sectornumber = SecNum(NewChild)
		NewChild_Fitness = FitnessFunction(drive,NewChild,sectornumber)	# Calculate the Fitness of new child
		Pop.append(Individual(NewChild,sectornumber,NewChild_Fitness))	
	#Reduce the size of Pop to PopSize
	SortPop()					#sort in decreasing order of fitness value
	RemoveDuplicate()			#Remove Duplicates
        if len(Pop)>PopSize:
            Pop=Pop[0:PopSize]#len(Pop)]#
        else:
	    Pop=Pop[0:len(Pop)]#PopSize]#len(Pop)]#

###################### Sort Pop ###############################
def SortPop():
	global Pop
	for i in range(0,len(Pop)):
		for j in range(i+1,len(Pop)-1):
			if (Pop[i].Fitness <= Pop[j].Fitness):
				X = Pop[i].Fitness
				Pop[i].Fitness = Pop[j].Fitness
				Pop[j].Fitness = X
				X = Pop[i].SectorNumber
				Pop[i].SectorNumber = Pop[j].SectorNumber
				Pop[j].SectorNumber = X
				X = Pop[i].Chromosome
				Pop[i].Chromosome = Pop[j].Chromosome
				Pop[j].Chromosome = X

def RemoveDuplicate():
	global Pop
	newPop=[]
	for i in range(0,len(Pop)-1):
		if Pop[i].Fitness == Pop[i+1].Fitness and Pop[i].SectorNumber == Pop[i+1].SectorNumber:
			continue
		newPop.append(Pop[i])
	newPop.append(Pop[i])
	Pop=newPop

def find(drive,ConsideredSectors):
	global Found
	global SectorEvaluated
	metainfo=open(AlgoName+'-Image-details'+".csv",'wa')
	clone=open(AlgoName+'Evidence_Image'+".img",'wa')
	for region in ConsideredSectors:
		print 'Processing of ',region, ' sector region started'
		i=long(region[0])
		j=long(region[1])
		SectorEvaluated = SectorEvaluated + (j-i+1) #for counting total accessed sectors for TP, TN, FP and FN calculations
		if j>=SectorCount:
			j=SectorCount
		for k in range(i,j):#while i<=j:#end:
			drive.seek(Start)
			sectr=long(k)*long(SectorSize)
			drive.seek(sectr)
			data=drive.read(SectorRead)
			md5hash=hsh(data)
			if md5hash in MD5HashOfTargetFile and md5hash != 'bf619eac0cdf3f68d496ea9344137e8b': #fitvalue(data) != 0:
				print k,'-->',md5hash
				clone.write(data)
				metainfo.write(str(k)+'Sector ,'+str(sectr)+' byte '+'-----'+str(md5hash)+'\n')
				j=j+1 
				Found.append(k)
	metainfo.close()
	clone.close()
	if Found != []:
		print Found
	else:
		print "Sorry! The target data is either not present in this region or Try searching the left regions."

def fltr(X): 
	f=[]
        X=np.array(X)  
        mean_duration  = np.mean(X)
        std_dev_one_test = np.std(X)   
	for i in X:
		if abs(i - mean_duration) <= std_dev_one_test and abs(i + mean_duration) >= std_dev_one_test:
			f.append(i)
	if f == []:
		std_dev_one_test = 0
		return [np.mean(X),std_dev_one_test]
	else:
		return [np.mean(f),np.std(f)]


def meanshift(X):
	clu=[]
	centers = [[1, 1], [-1, -1], [1, -1]]
	X=np.array(X)
	###############################################################################
	# Compute clustering with MeanShift
	# The following bandwidth can be automatically detected using
	bandwidth = estimate_bandwidth(X, quantile=0.2)#, n_samples=500)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_
	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	print("number of estimated clusters : %d" % n_clusters_)
	###############################################################################
	# Plot result
	#plt.figure(1)
	#plt.clf()
	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
	    my_members = labels == k
	    #print my_members
	    cluster_center = cluster_centers[k]
	#    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
	#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
	    clu.append([X[my_members, 0].tolist(), X[my_members, 1].tolist()])
	#plt.axis([0, SectorCount, 0, UBOfFitness+1])
	#plt.xticks(fontsize = 15)
	#plt.yticks(fontsize = 15)
	#plt.suptitle('Estimated Number of Clusters: %d' % n_clusters_, fontsize=20)
	#plt.title('Estimated Number of Clusters: %d' % n_clusters_)
	#plt.xlabel('Sector Regions',fontsize=18)
	#plt.ylabel('Fitness Values',fontsize = 18)
	#plt.show()   # SHOW PLOT
	boundary = []
	for i in clu:
		boundary.append([min(i[0]),max(i[0])])
	boundary.sort()
	return boundary


if __name__ == '__main__':
	################################# Parsing the arguments ###############################################
	parser = argparse.ArgumentParser(description='Differential Evolution (DE) based storage drive relevant data region identification and target data search for fast forensic investigation. \n Please input the target file or data.')
	parser.add_argument('-d', '--drive', required=True,help='The device ID of the drive. Eg. /dev/sda1')
	parser.add_argument('-o', '--outputDir', required=True,help='The directory to which DE results should be saved')
	parser.add_argument('-P', '--Population', required=False, type=int, default=100, help='The size of tye population for DE process. Default value is 100')
	parser.add_argument('-R', '--runs', required=False, type=int, default=5, help='The number of runs the DE process is executed. Default value is 10')
	parser.add_argument('-T', '--iteration', required=False, type=int, default=50, help='Total number of iteration per RUN. Default value for iteration is 200.')
	parser.add_argument('-X', '--crossover', required=False, type=float, default=0.5, help='The individual cross-over rate for population generation. Default value is 0.5')
	parser.add_argument('-Y', '--mutation', required=False, type=float, default=0.5, help='The chromosome mutation rate for population generation. Default value is 0.5')
	parser.add_argument('-N', '--filename', required=False, type=str, help='File name that need to be searched for analysis.')
	args = parser.parse_args()
	if not os.access(args.drive, os.W_OK):
		sys.exit("Unable to locate the storage drive %s" % (args.drive,) )
	if not os.path.isdir(args.outputDir) or not os.access(args.outputDir, os.W_OK):
		sys.exit("Unable to write to output directory %s" % (args.outputDir,) )
	if args.runs < 1:
		sys.exit('Number of runs must be greater than 0') 
	if args.iteration < 1:
		sys.exit('Number of iteration must be greater than 0')  
	if args.crossover <= 0 :
		sys.exit('Cross-over rate must be greater than 0')  
	if args.mutation <= 0:
		sys.exit('Mutation rate must be greater than 0')   
        ############################ PARSING END #######################################

        ################################# Setting-Up Arguments  ###############################################
        StorageID 	= args.drive
	CR		= args.crossover
	F		= args.mutation
	PopSize 	= args.Population
	Runs		= args.runs
	Iterations	= args.iteration
	AlgoName 	= args.outputDir+AlgoName+'-'
	Target		= args.filename
        ResultFileName 	= args.outputDir+ResultFileName#os.path.join(args.outputDir,ResultFileName)
	if not os.path.isfile(Target):
		sys.exit("Unable to find the given file %s or the Entropy value is invalid.\n Make valid entries." % (args.filename,) )
	f2=open(AlgoName+'-Target File-details'+".csv",'wa')
	c=0
	Fitval = []#Mean
	ddmean=[]#for mean
	with open(Target, 'rb') as f1:#(args.filename, 'rb') as f1:
		TargetProcessingTime = time.time()
       		while True:
       			b = f1.read(SectorRead)
       			if not b:
       				break
			FV = fitvalue(b) #Entropy Values
			Fitval.append(FV)
                        #print FV
			TargetFileED.append(FV)
			H = hsh(b)#Hash Values
			MD5HashOfTargetFile.append(H)
			f2.write(str(c)+','+str(FV)+','+str(H)+"\n")
			c=c+1
		print 'Time for processing target data: ',time.time()-TargetProcessingTime
		f2.close()
		f1.close()
	[TargetDataEDMean,TargetDataEDStDev]=fltr(TargetFileED)
	print "Mean and Std Dev.:",TargetDataEDMean," and ",TargetDataEDStDev, 'Max:', TargetDataEDMean+TargetDataEDStDev, 'Min:',TargetDataEDMean-TargetDataEDStDev
	LBOfFitness=TargetDataEDMean-TargetDataEDStDev
	UBOfFitness=TargetDataEDMean+TargetDataEDStDev
        hdd_parameters 	=''.join(["hdparm -g ",args.drive])     #Extract The Total Sector Number
        status,commandout = commands.getstatusoutput(hdd_parameters)
	# IF status=0 the this is ok and if status!=0 
	if status==0:
        	parameternumber = re.findall(r'\d+', commandout)
        	size = len(parameternumber)
        	start = int(parameternumber.pop())
        	SectorCount = int(parameternumber.pop())
		if len(parameternumber)>3:
			parameternumber.pop(0)
        	Dim = len(parameternumber)
        	for i in range(0,Dim):
			UB.append(int(parameternumber[i]))
			LB.append(0) 
	elif status!=0:
		print "Unable to detect the required parameters. \n Please provide the required parameters manually."
		op = int(raw_input("Enter 1 to Continue or 0 to Exit: "))
		if op==0:
			exit(0)
		else:
			SectorCount = int(raw_input("Enter total number of sectors: "))
			UB.append(int(raw_input("Enter Cylinder count: ")))
			UB.append(int(raw_input("Enter Head count: ")))
			UB.append(int(raw_input("Enter Sector/Track count: ")))
			LB.append(0)
			LB.append(0)
			LB.append(0)
			Dim = len(UB)
        ####################################### Processing  ###################################################
	file1 = open(ResultFileName,'wa')
	main(file1)
	file1.close()
	print "Data Exists in:", DataExists
