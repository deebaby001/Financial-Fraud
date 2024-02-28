# Financial-Fraud
An investigation of financial fraud investigation of a dataset

EDA and the Data Science Process
Data Science is an all encompassing area of science derived from various elements in the study of Science and Technology. When understanding
Exploratory Data Analysis (EDA) this is a part of the Data Science process. In the most simplistic terms, it involves, exploration of data and the analysis of the data. 
For the purposes of this project, the EDA consists of: Obtainding a viable dataset, Reviewing the dataset, Cleaning the dataset, Analyzing the data for the appropriate model, Fitting the model to the data and a final analysis--- How well did we do? And what are the Next Steps? in the evolution of this project and the overall Data Science process.

<!DOCTYPE html>  
<html>  
 <body>  
      <h3>Hypotheseis: This data set contains transaction data with 6,362,620 rows and 11 columns, including information about the transaction step, type, amount, origin and destination balances, fraud status, and flagged fraud status. 
          Assumption: It can be used to analyze patterns, detect fraudulent transactions, and develop predictive models for fraud prevention. </h3> 
 <body>  
</html>

<!DOCTYPE html>  
<html>  
 <body>  
      <h3>Research</h3> 
 <body>  
</html>

Research Goals
> Goal 1: Comparative Analysis of specific variables and how they interrelate and effect one another.
>
Variables
> The Attributes for analysis:( Fields) step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest  isFraud, isFlaggedFraud
> 
#1. DATA

The data is being furnished by an authored contributor and researcher by: (List information from original source)
Using AWS as a tool to access the 481KB file. 
The location being accessed is: https://s3.amazonaws.com/dstkh.datasets/PS_20174392719_1491204439457_log.csv

NOTE: Due to the size of the dataset I am using .gitignore to mask some of the .csv files from being loaded to the github repo.

#2. METHOD

The primary objective is the begin with clean data. Then place that data into a predictive model. The models being used will be:
1. Random Forest Decision Trees
2. HeatMap


#3. CLEANING REPORT

# Cleaning Report: Financial Fraud Dataset

## Introduction
The purpose of this report is to document the data cleaning steps performed on the Financial Fraud Dataset. The dataset includes order details such as:  customer IDs, product names, quantities, and prices.

## Data Overview
- **Financial Fraud**: Financial Fraud
- **Number of Rows**: 6,362,620 rows 
- **Number of Columns**: 11

## Data Cleaning Steps

### 1. Handling Missing Values
- N/A

### 2. Outlier Detection
- Detected outliers; removed extreme values
- Decision: 

### 3. Data Transformation
- Removed 2 columns: step and nameOrig


## Justifications
- **Missing Values**:N/A because this data possibly was preprocessed from bank.Depicted by the linearity that exists.
- **Outliers**: Removing extreme values
- **Data Transformation**: In this case the "step" and the "nameOrig" is not applicable to the projected positive overall performance.

## Conclusion
The cleaned dataset is now ready for further exploration and modeling. The documented steps ensure transparency and reproducibility.

The purpose of this is to present the data in the most unencumbered manner possible. The data has been cleared of duplicate rows, and columns, NAN, Non-NaN, and Blanks.

#4. EDA

1st: Pre-Processing 

The EDA Analysis Report as follows:
Attempted to run dataset; it took 15m 58.6 s at time of last run. This is intensive CPU usage on a basic laptop.

Description of Dataset Column Information and Type Information

Step: The "Step" is intended to represent a TimeStamp of each financial transactions represented in the dataset
Type: The "Type" of financial transaction, represented by: TRANSFER, CASH_IN, CASH_OUT, PAYMENT
Amount: The "Amount" of each financial transaction represented in US Dollars ($)
nameOrig: The "nameOrig" the Name of the origin financial account
oldbalanceOrg: The "oldbalanceOrg", represents the old balance of the origin account
newbalanceOrig: The "newbalanceOrig", represents the new balance of the origin account
nameDest: The "nameDest", represents the name of the destination account
oldbalanceDest: The "oldbalanceDest", represents the old balance of the destination account
newbalanceDest: The "newbalanceDest", represents the new balance of the destination account
isFraud: The "isFraud", represents that the account has been detected to be fraudulent
is FlaggedFraud: THe "is FlaggedFraud", represents that the account is selected for possible fraud 

Data Set TYPE Information
step              0
type              0
amount            0
nameOrig          0
oldbalanceOrg     0
newbalanceOrig    0
nameDest          0
oldbalanceDest    0
newbalanceDest    0
isFraud           0
isFlaggedFraud    0
dtype: int64

Description of DataSet

CODE: df.describe
<bound method NDFrame.describe of          step      type      amount     nameOrig  oldbalanceOrg  \
0           1   PAYMENT     9839.64  C1231006815      170136.00   
1           1   PAYMENT     1864.28  C1666544295       21249.00   
2           1  TRANSFER      181.00  C1305486145         181.00   
3           1  CASH_OUT      181.00   C840083671         181.00   
4           1   PAYMENT    11668.14  C2048537720       41554.00   
...       ...       ...         ...          ...            ...   
6362615   743  CASH_OUT   339682.13   C786484425      339682.13   
6362616   743  TRANSFER  6311409.28  C1529008245     6311409.28   
6362617   743  CASH_OUT  6311409.28  C1162922333     6311409.28   
6362618   743  TRANSFER   850002.52  C1685995037      850002.52   
6362619   743  CASH_OUT   850002.52  C1280323807      850002.52   

         newbalanceOrig     nameDest  oldbalanceDest  newbalanceDest  isFraud  \
0             160296.36  M1979787155            0.00            0.00        0   
1              19384.72  M2044282225            0.00            0.00        0   
2                  0.00   C553264065            0.00            0.00        1   
3                  0.00    C38997010        21182.00            0.00        1   
4              29885.86  M1230701703            0.00            0.00        0   
...                 ...          ...             ...             ...      ...   
6362615            0.00   C776919290            0.00       339682.13        1   
6362616            0.00  C1881841831            0.00            0.00        1   
6362617            0.00  C1365125890        68488.84      6379898.11        1   
6362618            0.00  C2080388513            0.00            0.00        1   
6362619            0.00   C873221189      6510099.11      7360101.63        1   

         isFlaggedFraud  
0                     0  
1                     0  
2                     0  
3                     0  
4                     0  
...                 ...  
6362615               0  
6362616               0  
6362617               0  
6362618               0  
6362619               0  

[6362620 rows x 11 columns]>

Technique used to Sample the original Dataset

Due to the size of the dataset, I employed the sampling technique using,
CODE:  sample_df = df.sample(n=1000000)
which yielded:
CODE: sample_df.shape
RESULTING: (1000000, 11) 

#5. MODELING

Final: Model Data Visualization with Written Analysis

#7. FURTHER RESEARCH
Interesting Facts Discovered: There were 16 rows that were unflagged as "isFlaggedFraud", meaning that there were set as "isFraud"


#8. CREDITS
Original Project was provided by: (See data from Canvas site per instructor. Ensemble Methods For Financial Fraud Detection)
The data that was accessed for this project is located at the following sites: 'https://s3.amazonaws.com/dstkh.datasets/PS_20174392719_1491204439457_log.csv'