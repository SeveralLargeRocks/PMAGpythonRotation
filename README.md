# PMAGpythonRotation
De-rotation, visualisation and processing of palaeomagnetic poles and plate circuits. 

Basic workflow is Curator to Calculator to either Plotter or Plate Circuit Segmenter. The curator, calculator and plotter are found in cur_calc_plot and the plate circuit analysis folder has Plate_Circuit_Segmenter_With_Bram_Code.ipynb which is the currently working placte circuit analysis.

Loads of stuff is either now redundant, duplicate or is badly stored in general.

Lots of the actual code is also now redundant or is not working fully, needs a general clean up and standardisation before I am going to work on it further. 

TO-DO list:
-Revise information displayed during curator function
-Check dropping of poles even when minimum filtering is applied during curator
-More info for Calculator step
-Fix allocations of R values for kriging stepo
-For map creation make the display of kriging/voronoi be an option during creation not just hard coded in
-Map creation druing plotter is VERY BROKEN
-
