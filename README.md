# ecgScorer-Python
A Python port of the MATLAB [ecgScorer](https://github.com/ecgScorer/ecgScorer) toolbox. Currently, it only includes the ECG Quality Assessment functionality.

## Usage
You can load your ECG signal from a TXT, CSV, or XLSX file and compute the quality scores.
There is a `signal_test.txt` file in the `Example` folder to test the software.

> "All algorithms must be used with ECGs as standing vectors or matrices with leads columnwise arranged (temporal dimension in lines)"  — original ecgScorer README
