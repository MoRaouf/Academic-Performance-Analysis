# Retail-Sales-Forecasting

End-to-end project of a web application that predicts weekly sales for a retail store. It uses a machine learning model that takes into account various features about a retail store to make the prediction. The web application was deployed in two forms, a Docker image & a web app on AWS Elastic Beanstalk.

**The deployed app on AWS Elastic Beanstalk is accessible from this link: http://retailsales-env.eba-r7zr85ez.us-east-1.elasticbeanstalk.com/**

<!-- Add Video here !!!!!!!!!!!!!!!!! -->

Problem Statement
---
The objective of this project is to develop a machine learning model to predict the weekly sales of various departments in different stores of a retail company. The model will be trained on historical sales data along with other relevant features such as markdowns, CPI, unemployment rates, and more. The goal is to develop an accurate predictive model that can assist the company in making informed decisions about store and department-level operations, such as staffing, inventory management, and marketing strategies.


Dataset
---
The dataset used for this project is the Walmart Recruiting - Store Sales Forecasting dataset, which can be found on Kaggle. The dataset contains historical sales data for 45 Walmart stores located in different regions, along with other relevant features such as holidays, temperature, fuel prices, and markdowns. The objective of this project is to predict the weekly sales of each department in each store.

The training & testing datasets consist of weekly sales data for the period from February 5, 2010 to October 26, 2012, while the prediction dataset covers the period from October 27, 2012 to December 07, 2013.



Installation
---
* Clone this repository:
    ```
    git clone https://github.com/MoRaouf/Retail-Sales-Forecasting.git
    ```
* Set up the virtual environment and all required dependencies by:
  * Setting up a `python=3.8` virtual environment
  * run: `pip install -r requirements.txt`

* Change directory & run Flask app:
    ```
    cd Retail-Sales-Forecasting
    python application.py
    ```
* Open a web browser and go to http://localhost:5000 to access the application.

* Enter the required input data and click on the "Predict Weekly Sales" button to get the predicted weekly sales for the selected parameters.

Deployment to AWS Beanstalk
---
The application was deployed to AWS Beanstalk. [Access it here](http://retailsales-env.eba-r7zr85ez.us-east-1.elasticbeanstalk.com/).


Deployment as Docker image
---
* To use the app locally:
    1. Pull the image from Docker Hub then run a container from it:
        ```
        docker image pull moraouf/sales-forecast:v1
        docker container run -it --name sales-forecast-app -p 5000:5000 moraouf/sales-forecast:v2
        ```
    2. Run flask app from http://localhost:5000/


* To deploy your app as a Docker image, build the image from the `Dockerfile` using the following command:
    ```
    docker image build -t sales-forecast .
    ```
<!-- 2. Log in to your Docker Hub account from the CLI.
```
docker login
```
3. Rename the image & push it to your Docker Hub account.
```
docker image tag sales-forecast moraouf/sales-forecast:v1
docker image push moraouf/sales-forecast:v1
``` -->


Requirements
---
This project used the following libraries:
```
numpy
pandas
matplotlib
seaborn
sklearn
xgboost
catboost
flask
pyyaml
```

Contributing
---
If you would like to contribute to this project, please open an issue or submit a pull request.

<!-- License
---
This project is licensed under the MIT License. See the LICENSE file for more information. -->
