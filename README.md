# Water Quality Machine Learning Application

## Overview
This project aims to develop a machine learning application for water quality analysis. The application utilizes various machine learning algorithms to predict water quality based on input parameters such as pH level, dissolved oxygen, and turbidity. It also includes features for data visualization, model training, and evaluation.

## Table of Contents
1. [Features](#Features)
2. [Pre-requisites](#pre-requisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [Acknowledgements](#acknowledgements)

## Features
- Data Preprocessing: The application preprocesses raw water quality data, including handling missing values and scaling features.
- Model Training: It trains machine learning models using various algorithms such as LSTM (Long Short-Term Memory) and others to predict water quality.
- Evaluation Metrics: The application provides evaluation metrics such as accuracy, precision, recall, and F1-score to assess the performance of trained models.
- Data Visualization: It offers visualizations of water quality data using libraries like Matplotlib and Seaborn to gain insights into the dataset stores the gerenated plots in a folder called plots with the timestamp.
- Google Sheets Integration: The application integrates with the Google Sheets API to facilitate data import/export and collaboration.
- Email Updates: The Water Quality Reports arre sent via email address given when registering to your account

## Pre-requisites
- Python version - `3.9.0`
- Mysql version - `8`
- Google sheets api - follow this tutorial along to get Google sheets api
```bash 
https://youtu.be/P16uA1Hl4DI?si=yhVFReLNcDGe7_rC
```
- Sheet ID - Create a new google sheet 
https://docs.google.com/spreadsheets/d/abc1234567/edit#gid=0 is "abc1234567".
- Mail Setup -follow this tutorial along to get App password
```bash 
https://youtu.be/hXiPshHn9Pw?si=snMjPPynOttla-UF
```

## Installation
1. Clone the repository: 
```bash
git clone https://github.com/20481A5450/WaterQuality-MLApp.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up MySQL database and configure the settings in
```bash
settings.py
```
4. Refer the DB.txt file and create the tables accordingly & Run migrations:
```bash
python manage.py migrate
```
5. Start the development server:
```bash
run.bat
```

## Usage
1. Upload Water Quality Data: Import water quality data into the application.
2. Preprocess Data: Preprocess the data to handle missing values and scale features.
3. Train Models: Train machine learning models using different algorithms.
4. Evaluate Models: Evaluate the performance of trained models using various evaluation metrics.
5. Visualize Data: Visualize water quality data to gain insights and identify patterns.
6. Email Updates: Water quality reports are sent to the registered email address.

## Contributing
Contributions are welcome! Feel free to open issues for bug fixes, feature requests, or any improvements you'd like to propose.
Feel Free to raise an issue if you face any difficulty in any phase.

## Acknowledgements
- Thanks to the contributors who have helped in developing and testing this application.
- Special thanks to the open-source community for providing valuable libraries and resources.
