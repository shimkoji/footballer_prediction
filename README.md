# Young Footballer's Future Skill Prediction

## Overview

In the competitive world of football, identifying and nurturing promising young talent is crucial for clubs' long-term success. This project presents a machine learning system designed to predict a young footballer's overall skill level five years into the future, providing valuable insights for talent scouts and club management.

## Project Description

This system leverages machine learning techniques to forecast the future potential of young footballers. By utilizing the comprehensive [FIFA dataset](https://sports-statistics.com/sports-data/fifa-2022-dataset-csvs/), we've developed a robust prediction model that can estimate a player's overall skill rating five years from now.

### Key Features

1. **Data Source**: We use the FIFA dataset, which provides detailed attributes for thousands of players across multiple years.
2. **Target Group**: The system focuses on young players aged 15-22, capturing the crucial developmental years of a footballer's career.
3. **Multiple Model Approach**: We train and compare three state-of-the-art machine learning models:
   - XGBoost
   - LightGBM
   - Random Forest
4. **Historical Data Analysis**: The system trains on player attributes from FIFA 15 and correlates them with the same players' overall skills in FIFA 19, providing a realistic 5-year projection.

## How It Works

1. The system extracts relevant features for young players (aged 15-22) from the FIFA 15 dataset.
2. It then matches these players with their corresponding data in FIFA 19 to create a labeled dataset.
3. The three machine learning models are trained on this data, learning the relationships between initial attributes and future overall skill.
4. The best-performing model is selected for making predictions on new data.

## Usage

To use this system and predict a young player's future skill, follow these steps:

1. Ensure you have Docker installed on your system.
2. Clone this repository to your local machine.
3. Open a terminal and navigate to the project directory.
4. Run the following commands:

```bash
docker-compose up --build
make build
make predict