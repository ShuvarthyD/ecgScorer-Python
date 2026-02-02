# ecgScorer-Python
A Python port of the MATLAB [ecgScorer](https://github.com/ecgScorer/ecgScorer) toolbox. Currently, it only includes the ECG Quality Assessment functionality.

## Usage
You can either use the `scorer12` function in the ECGScorer script or use the graphical interface by running the GUI. You can load your ECG signal from a TXT, CSV, or XLSX file and compute the quality scores.  
Note that the data must follow the format quoted from the original ecgScorer README:
> "All algorithms must be used with ECGs as standing vectors or matrices with leads columnwise arranged (temporal dimension in lines)"
There is a `signal_test.txt` file in the `Example` folder to test the software.
