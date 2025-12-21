Requirements before running: (All the below packages being in the latest release will be better for the )
1. Python (3.11.4)
2. Pandas (2.3.3)
3. Numpy (1.26.0)
4. Sklearn (1.8.0)
5. Matplotlib (3.10.8)
6. Tensorflow (2.13.0)


The following will show you how to run my end of the code, and to replicate the results

1. Download the onedrive files for the dataset (named and linked as Dataset in page 2 of the report).
2. After which run files 01-01 and 01-02 for merging the individual files into the final dataset for further processing. (Approximate run time 30-60 mins based on CPU)
3. For the models run files 02-01-01, 02-02-01, 02-02-02, 02-03-01 and 02-04-01.

     a. Please note we used a primitive batch implementation, hence if there is error while running that block of code (the model is trained with most of the data, apart from the last few datapoints), please proceed to the next accuracy block.
   
     b. When running 02-02-01 and 02-02-02 (SVM implementation), we have reduced the number of datapoints to keep runtime manageable (approximately 1-2 hours per file).
   
     c. File 02-02-03 was a work in progress, where we chose specific hyperparameters to run for a bigger subset of the data, but each model took approximately greater than 2 hours to run, hence we only ran a few models and stopped as it was taking too long to run.


If you have any issues running any part of my code, do let me know via moodle, or email.
