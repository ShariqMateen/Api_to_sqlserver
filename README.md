# ETL Pipeline Project

## Overview

This project implements an ETL (Extract, Transform, Load) pipeline using Python. It extracts data from the DGEFP API, transforms it with pandas, and loads it into an SQL Server database. The script includes data validation, logging, and scheduled execution, efficiently handling both updates and new records.

## Features

- Extracts data from the DGEFP API (`liste-publique-des-of-v2` dataset).
- Cleans and transforms data using pandas (e.g., handling missing values, type conversions).
- Loads data into an SQL Server database with versioning (active/inactive records).
- Implements logging for debugging and monitoring.
- Supports scheduled execution (commented out for daily runs at 2 AM).

## Prerequisites

- Python 3.8+
- SQL Server instance (e.g., `localhost\\SQLEXPRESS`)
- Required Python libraries:
  - `requests`
  - `pandas`
  - `pyodbc`
  - `numpy`
  - `schedule`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/ShariqMateen/Api_to_sqlserver.git
   cd Api_to_sqlserver
   ```
2. Install the required Python libraries:

   ```
   pip install requests pandas pyodbc numpy schedule
   ```
3. Set up your SQL Server database:
   - Create a database named `Organization`.
   - Update the SQL Server connection settings in `Last.py` (e.g., `SQL_SERVER`, `SQL_USERNAME`, `SQL_PASSWORD`).

## Usage

1. Ensure your SQL Server is running and accessible.
2. Run the script:

   ```
   python Last.py
   ```
3. The script will:
   - Fetch data from the API.
   - Clean and transform the data.
   - Load it into the `training_organizations` table in SQL Server.
   - Log the process in `training_data.log`.

## Scheduling

To enable daily scheduled runs (e.g., at 2 AM):

1. Uncomment the scheduling code in `Last.py`:

   ```python
   schedule.every().day.at("02:00").do(run_daily_job)
   ```
2. Uncomment the `while` loop to keep the scheduler running:

   ```python
   while True:
       schedule.run_pending()
       time.sleep(60)
   ```
3. Run the script as a background process.

## Logging

- Logs are saved to `training_data.log` and printed to the console.
- Log levels: INFO, DEBUG, ERROR.

## Database Schema

The script creates a table `training_organizations` in the SQL Server database if it doesn't exist. Key columns include:

- `siretetablissementdeclarant`: Unique identifier (with `is_active` constraint).
- `denomination`, `siren`, `adressephysiqueorganismeformation_*`: Organization details.
- `certifications_*`: Boolean flags for certifications.
- `informationsdeclarees_*`: Training statistics.
- `start_date`, `end_date`, `is_active`: Record versioning.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.
