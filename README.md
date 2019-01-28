# DEBasedTargetSearch

# Manual for the tool:

python DE_Based_Target_Search.py -h
usage: DE_Based_Target_Search.py [-h] -d DRIVE -o OUTPUTDIR [-P POPULATION]
                                 [-R RUNS] [-T ITERATION] [-X CROSSOVER]
                                 [-Y MUTATION] [-N FILENAME]

Differential Evolution (DE) based storage drive relevant data region
identification and target data search for fast forensic investigation. Please
input the target file or data.

optional arguments:
  -h, --help            show this help message and exit
  -d DRIVE, --drive DRIVE
                        The device ID of the drive. Eg. /dev/sda1
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        The directory to which DE results should be saved
  -P POPULATION, --Population POPULATION
                        The size of tye population for DE process. Default
                        value is 100
  -R RUNS, --runs RUNS  The number of runs the DE process is executed. Default
                        value is 10
  -T ITERATION, --iteration ITERATION
                        Total number of iteration per RUN. Default value for
                        iteration is 200.
  -X CROSSOVER, --crossover CROSSOVER
                        The individual cross-over rate for population
                        generation. Default value is 0.5
  -Y MUTATION, --mutation MUTATION
                        The chromosome mutation rate for population
                        generation. Default value is 0.5
  -N FILENAME, --filename FILENAME
                        File name that need to be searched for analysis.

# Sample Syntax
python DE_Based_Target_Search.py -d /dev/sd? -o <Output Ditectory> -T 30 -Y 0.40 -X 0.58 -R 20 -P 200 -N <Target File Location>
