# Heart disease predictions

## Data source

The 2020 BRFSS data continue to reflect the changes initially made in 2011 for weighting methodology (raking) and adding cell-phone-only respondents. The aggregate BRFSS combined landline and cell phone data set is built from the landline and cell phone data submitted for 2020 and includes data from 50 states, the District of Columbia, Guam, and Puerto Rico

## About Dataset (from [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease))

### Key Indicators of Heart Disease

2020 annual CDC survey data of 400k adults related to their health status

What topic does the dataset cover?

According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans, American Indians and Alaska Natives, and white people).

About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking.

Other key indicator include diabetic status, obesity (high BMI), not getting enough physical activity or drinking too much alcohol.

Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare.

Computational developments, in turn, allow the application of machine learning methods to detect "patterns" from the data that can predict a patient's condition.

### Where did the dataset come from and what treatments did it undergo?

Originally, the dataset come from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents.

As the CDC describes: "Established in 1984 with 15 states, BRFSS now collects data in all 50 states as well as the District of Columbia and three U.S. territories.

BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.". The most recent dataset (as of February 15, 2022) includes data from 2020.

It consists of 401,958 rows and 279 columns. The vast majority of columns are questions asked to respondents about their health status, such as "Do you have serious difficulty walking or climbing stairs?" or "Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]".

In this dataset, I noticed many different factors (questions) that directly or indirectly influence heart disease, so I decided to select the most relevant variables from it and do some cleaning so that it would be usable for machine learning projects.

### What can you do with this dataset?

As described above, the original dataset of nearly 300 variables was reduced to just about 20 variables.

In addition to classical EDA, this dataset can be used to apply a range of machine learning methods, most notably classifier models (logistic regression, SVM, random forest, etc.).

You should treat the variable "HeartDisease" as a binary ("Yes" - respondent had heart disease; "No" - respondent had no heart disease).

But note that classes are not balanced, so the classic model application approach is not advisable.

Fixing the weights/undersampling should yield significantly betters results.

## Methods that will be used

### First preview of data with a Decision tree

### Advanced implementation of a random forest

### Advanced implementation of a gradient boosted decison tree

## Results comparison between gradient boosted decision tree and random forest

## Research of other advanced technics if results don't match expectations

## Run the code

```bash
conda create --name heart-disease-recognition python=3.9
```

```bash
conda activate heart-disease-recognition
```

```bash
nvidia-smi
```

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.8.0
```

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
```

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

```bash
python3.9 -m pip install --upgrade pip
```

```bash
python3.9 -m pip install tensorflow==2.12.0
```


```bash
python3.9 -m pip install tensorflow_decision_forests
```

```bash
python3.9 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
