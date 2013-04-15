
We've provided several sample data that we've parsed from the MSD (Million Song Database).
=======
************
** README **
************
This project focuses on the grouping of songs based on meaningful and measurable attributes of songs found on the EchoNest via the Million Song Database by parallelizing isomaps on each tracks. The process is divided into three main parts:
1) Data extraction & processing
2) Applying isomap on the data
3) Visualization and extraction

***********
** FILES **
***********
The following files are included in this submission:
- songs_parallel.py
- songs_serial.py
- mrjob_config_file.txt
- tiny.dat
- out100.dat
- out1000.dat
- out10000.dat
- isomap_parallel.py
- driver.py
- driver_parallel.py
- visualization.py

**********************************
** Data Extraction & Processing **
**********************************
If you would like to test it on a small subset of the online data, use tiny.dat as your input file.

In order to extract the data serially locally, run:
$ python songs_serial.py [input file] > [output file]

In order to extract the data in parallel locally, run:
$ python songs_parallel.py [input file] > [output file]

To set up EMR, edit mrjob_config_file.txt to include credentials, then run:
$ export MRJOB CONF=/home/you/yourpath/mrjob_config_file.txt

In order to extract the data in parallel on Amazon EMR, run:
$ python songs_parallel.py -r emr [input file] > [output file]

In order to extract the full data on Million Song Database, run:
$ python songs_parallel.py -r emr 's3://tbmmsd/*.tsv.*' > [output file]

This was done to produce out100.dat, out1000.dat, and out10000.dat, with 100, 1000, and 10000 songs respectively.

*********************
** Applying Isomap **
*********************


This part of the project implements the isomap algorithm to the rows of song data pulled from the million song database.

For comparison, we've implemented both the serial and parallel version of the isomap with the serial version being run by the command:

$ python driver.py

and the parallel version being run by the command:

$ python driver_parallel.py

The input file for analysis can be specified in the driver.py and driver_parallel.py files respsectively.

********************************
** Visualization & Extraction **
********************************

visualization.py processes and produces plots for the input data file, which consists of the output file from the isomap algorithm above (python drive_parallel.py) and the input file for driver_parallel.py (zipping together the isomap scores and the associated track data for plots)

It can run for various data length outputs - input files and number for 'TRACKS' must be adjusted in the file, but visualization.py runs for the largest data set since it produces the most useful visualizations.

simply run:

$ mpirun -n 4 python visualization.py 

runs with 2, 4 or 8 instances

Output: 5 plots. To proceed through the code running, close each graph to proceed to the next one
