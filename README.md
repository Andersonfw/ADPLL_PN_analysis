# ADPLL_PN_analysis
Python Scripts to analyse the PN of the ADPLL


## How to use

Run the file analyses.py and edit the variables based on the files on the "data" folder. The options are:

FREQ = 2402 or 2440 or 2418123 or 2480
CORNER = WORST or TYP or BEST
SETTING = FPREDICT or FPREDICT_SDM1 or FPREDICT_SDM2
TDC_DCO_CORNER =  WORST or TYP or BEST
### Example 
Simulations with a frequency desired at 2.402 GHz, worst corner, and with FPREDICT and SDM2 enable and with TDC and DCO model running on a TYP corner.

FREQ = 2402 
CORNER = WORST
SETTING = FPREDICT_SDM2
TDC_DCO_CORNER =  TYP

The files data path is: data\TYP\2402_TYP\FPREDICT_SDM2
