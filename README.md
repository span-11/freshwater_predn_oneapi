
# Machine Learning tool to predict the quality of freshwater

The project is based on one of the themes Intel® oneAPI Hackathon for Open Innovation


## Machine Learning Challenge Track: Predict the quality of freshwater

### Problem:

Freshwater is one of our most vital and scarce natural resources, making up just 3% of the earth’s total water volume. It touches nearly every aspect of our daily lives, from drinking, swimming, and bathing to generating food, electricity, and the products we use every day. Access to a safe and sanitary water supply is essential not only to human life, but also to the survival of surrounding ecosystems that are experiencing the effects of droughts, pollution, and rising temperatures.

### Expected Solution:

In this track of the hackathon, we apply Machine learning concepts and leverage oneAPI capabilities to help global water security and environmental sustainability efforts by predicting whether freshwater is safe to drink and use for the ecosystems that rely on it. 

## Objective 

The objective of this project are two fold:

a. Create a prediction model which can determine whether the target source is a freshwater or not.

b. Make the model accessable to end users, and aid them solve real world problem 
## Approach : Constructing the prediction model

### Data Audit & Analysis 

Usage of OneAPI library modin helped accelerate the time of data load and optimisation. 

The input data had 'Target' as the dependent and the below listed independent variables. 

a. Continuous Attributes: pH, Iron, Nitrate,Chloride, Lead, Zinc, Color, Turbidity, Fluoride, Copper, Odor, Sulfate, Conductivity, Chlorine, Manganese, Total Dissolved Solids, Source,Water Temperature, Air Temperature

b. Categorical Attribute: Day,Time of Day, Month, Source, Color

The attributes 'Day', 'Month' and 'Time of Day' were not considered in order to avoid over-fitting and make the model generic to predict out of time data. 

Attribute 'Source' was transformed into continuous variable using one-hot encoding. On the other hand the attribute 'Color' was converted from nominal to ordinal with following encoding: Colorless:1,Near Colorless:2,Faint Yellow:3,Light Yellow:4,Yellow:5
 

### Feature Engineering 

The dataset was balanced using imblearn. Boosting algorithm fared better, and within them LightGBM fared the best. The run time was accelerated by using sklearn library of OneAPI tool

![image](https://user-images.githubusercontent.com/122376420/221431972-1906c422-1b22-45fd-8a1c-20bc726d58bf.png)


From the set of 22 input features were able to drill down to 14 primary attributes needed for the prediction : Chloride,Chlorine,Color,Copper,Fluoride,Iron,Manganese,Nitrate,Odor,Sulfate,Total Dissolved Solids,Turbidity,Zinc,pH

### Optimisation and final model 

Using daal4py we were able to gain a faster execution time of LightGBM and derive the model to predict the target source (whether it is a freshwater or not)
## Usage & Implementation

### Usage 

The model was deployed in cloud and API was created to enable end users to leverage the prediction model to predict whether a target source is freshwater. A user can access the prediction model on this URL: https://freshwater-pred-38.azurewebsites.net/

### Implementation on local system 

The following steps underline the approach for implementation :

1. Go to the project directory in your local system. Activate the python version 3.8 environment

2. Clone the repository: Execute the following code 

```
git clone https://github.com/span-11/freshwater_predn_oneapi.git
```

3. Go to the repository folder 

```
cd freshwater_predn_oneapi
```

4. Install the needed libraries by executing below comands
```
pip install modin[all] os xgboost lightgbm catboost time warnings numpy seaborn matplotlib scikit-learn scikit-learn-intelex imblearn re pickle
```

```
pip install -r requirements.txt
```
5. Copy the data into the folder from this link (https://s3-ap-southeast-1.amazonaws.com/he-public-data/datasetab75fb3.zip). Unzip it. 

6. Open the ipynb file 'Code_FreshwaterPrediction.ipynb' and 

change the path in the ipynb file from 

```
C:/Users/sidha/OneDrive/Desktop/Work/DS/HackerEarth/Intel/Data/datasetab75fb3/dataset.csv
```

to the path of file in your system

7. Execute the python code and copy the model pickle file with the name 'final_model_lgbm.pkl' in the project directory 

8. Run the following command  
```
python app.py
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.
