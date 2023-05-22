# **Vietnamese ID Card Extractor (New ID Card Format)**

## **Introduction**

An web application helps us to extract information from Vietnamese chip-based ID card in a second. This application aims
to reduce human typing workload and saves more time.

## Prerequisites

- Install python 3.7 or later
- Clone this repository

## **Installation**

> **Warning**
>
> Enabling virtualenv is recommended

All requirement libraries are listed in requirements.txt. You can install it by using:

``` bash
pip install -r requirements.txt
```

## **Usage**

- Create an AWS account and set up authentication credentials for your account. (
  Read <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console">this</a>)
- Once the user has been created, create and retrieve the keys used to authenticate the user. (
  Read <a href = "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey">
  this</a>)
- Create ```sources/Controllers/config.py``` file and copy all values from ```sources/Controllers/config.exampe.py``` to
  it
- Replace your access_key_id and secret_access_key in the new config file
- Run ```python sources/Controllers/create_collection```

- Run the main file ``` python app.py ```

- Use ```localhost:8080``` to test all of the endpoints and enjoy! ( •̀ ω •́ )✧
