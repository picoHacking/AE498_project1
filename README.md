# AE498_project_1
Solar degradation analysis of NLR SERF-EAST solar power system

For quick setup, download the reduced dataset (pv_summary_2011_2023.csv, 7.67 KB) from this repository, and only run solar_degradation.py to see results.

To run solar_degradation.py and data_reduction.py, be sure to install all requirements listed in the requirements.txt file.

For manual data reduction:
Get a free API key at https://developer.nlr.gov/ (formerly NREL developer network) and set:
1. export NSRDB_API_KEY="your_key"
2. export NSRDB_EMAIL="you@example.com"
3. paste API key as a string into line 40 of data_reduction.py and un-comment the line
4. paste email into line 41 in the string "your_email"
5. run data_reduction.py and wait until the output file is generated (this may take several minutes)
