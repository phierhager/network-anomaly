# Network Anomaly Detection

This repository contains code and resources for detecting network anomalies using machine learning techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## Introduction
Network anomaly detection is crucial for identifying unusual patterns that may indicate security threats or performance issues. This project aims to provide a robust solution for detecting such anomalies in network traffic.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone git@github.com:phierhager/network-anomaly.git
cd network-anomaly
poetry install
poetry shell
```

## Datasets
To be able to train, download the UNSW-NB15 dataset from here (https://www.kaggle.com/datasets/dhoogla/unswnb15) or the CIC-IDS2017 dataset from here (https://www.kaggle.com/datasets/dhoogla/cicids2017) and save them in a data/ folder on top level.

## Usage
To train a detection model refer to the `train_tutorial.ipynb` notebook. The real-time package inspection is under development.