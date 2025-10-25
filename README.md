The directory 4\_switch\_data contains the RAW data for each switch in form of CSV files with tags whether a row in the file is REAL or FAKE.



The naming convention is : switch\_log\_<switch\_number>.csv



In our case, switch1 was misreporting. So switch\_log\_1.csv will contain some FAKE tag rows.



The file: misreport\_all\_datasets.zip contains separate directories for 4 different scenarios of attack parameters discussed in the paper, used to train the system.

The file: align-logs.py contains the code for aligning all raw logs for switches into a common dataset according to timestamps.

The file: augment-features.py contains the code for augmenting the aligned dataset and creating enhanced features for the dataset.


The file: training\_code.ipynb contains the training code.
