# ecgScorer-Python
A Python port of the MATLAB [ecgScorer](https://github.com/ecgScorer/ecgScorer) toolbox. Currently, it only includes the ECG Quality Assessment functionality.

## Usage
You can load your ECG signal from a TXT, CSV, or XLSX file and compute the quality scores. The data must follow the [required format](https://github.com/ecgScorer/ecgScorer#:~:text=All%20algorithms%20must%20be%20used%20with%20ECGs%20as%20standing%20vectors%20or%20matrices%20with%20leads%20columnwise%20arranged%20(temporal%20dimension%20in%20lines)).  
There is a `signal_test.txt` file in the `Example` folder to test the software.
