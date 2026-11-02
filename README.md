# AE498_project1
Solar degradation analysis of NLR SERF-EAST solar power system

For quick setup, download the reduced dataset from this repository, and only run solar_degradation.py to see results.

For manual data reduction:
Get a free API key at https://developer.nlr.gov/ (formerly NREL developer network) and set:
1. export NSRDB_API_KEY="your_key"
2. export NSRDB_EMAIL="you@example.com"
3. paste API key as a string into line 40 of data_reduction.py and un-comment the line
4. paste email into line 41 in the string "your_email"
5. run data_reduction.py and wait until the output file is generated (this may take several minutes)
