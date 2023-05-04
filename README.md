# Retail-Sales-Forecasting

Problem Statement
---


App
---


Dataset
---


Installation
---
* Clone this repository:
    ```
    git clone https://github.com/MoRaouf/Retail-Sales-Forecasting.git
    ```
* Set up the virtual environment and all required dependencies by:
  * Setting up a `python=3.8` virtual environment
  * run: `pip install -r requirements.txt`

* Change directory & run VSCode:
    ```
    cd Retail-Sales-Forecasting
    code .
    ```

Deployment to AWS Beanstalk
---

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
```
