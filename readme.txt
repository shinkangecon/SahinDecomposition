=======================================================================================================================
Replication files for:

Flow Origins of Labor Force Participation Fluctuations
Michael Elsby, Bart Hobijn, Fatih Karahan, Gizem Koşar, and Ayşegül Şahin
AEA - Papers and Proceedings, May 2019

Bart Hobijn, December, 2018
=======================================================================================================================

Directory structure

This program will only run if the files are extracted in the directory structure that they are compressed in

-----------------------------------------------------------------------------------------------------------------------

Software:

Python/Anaconda:
Dependencies listed in the file condapythonenv.txt

classes used in the /classes subdirectory to load data and make charts. This needs to be a subdirectory of the working
directory from which the program is run.

-----------------------------------------------------------------------------------------------------------------------

Program files:

Decomposition**.py, where ** denotes the gender for which results are calculated

-----------------------------------------------------------------------------------------------------------------------

Output:

The following subdirectories:

/xlsx    Workbooks with raw data and raw data and results

	 settings["Load"]["Fresh"] = True will update results to most recent data from BLS. These replication files are
	 using the vintage of the data that the paper is based on.

/tex     Latex tables with average stocks and flow rates by gender used to make Table1 in paper

/pdf     Directory with pdf versions of the charts

		**AreaChart###.pdf
		**Decomposition3a###.pdf

		are the particular charts used in the paper. Here ** stands for the gender and ### denotes the formatting 
		style _paper_serif is the one used in the paper.

/png     Directory with png bitmap version of figures

/svg	 Directory with svg scalable vector graphics version of figures

/html    Contains webcharts if settings[###]['Web']=True in the programs


