import requests
import pandas as pd
import pyodbc
import numpy as np
from datetime import datetime, date
import logging
import schedule
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "https://dgefp.opendatasoft.com/api/explore/v2.1/catalog/datasets/liste-publique-des-of-v2/exports/csv"

# SQL Server connection settings
SQL_SERVER = 'localhost\\SQLEXPRESS'
SQL_DATABASE = 'Organization'
SQL_USERNAME = 'testuser'
SQL_PASSWORD = '12345'
SQL_TABLE = "training_organizations"

# SQL CREATE TABLE Query
create_table_query = """
IF NOT EXISTS (
    SELECT * FROM sysobjects WHERE name='training_organizations' AND xtype='U'
)
BEGIN
    CREATE TABLE training_organizations (
    numerodeclarationactivite BIGINT,
    numerosdeclarationactiviteprecedent BIGINT,
    denomination VARCHAR(255),
    siren INT,
    siretetablissementdeclarant BIGINT,
    adressephysiqueorganismeformation_voie VARCHAR(255),
    adressephysiqueorganismeformation_codepostal INT,
    adressephysiqueorganismeformation_ville VARCHAR(100),
    adressephysiqueorganismeformation_coderegion INT,
    geocodageban FLOAT,
    certifications_actionsdeformation BIT,
    certifications_bilansdecompetences BIT,
    certifications_vae BIT,
    certifications_actionsdeformationparapprentissage BIT,
    organismeetrangerrepresente_denomination VARCHAR(255),
    organismeetrangerrepresente_voie VARCHAR(255),
    organismeetrangerrepresente_codepostal VARCHAR(50),
    organismeetrangerrepresente_ville VARCHAR(100),
    organismeetrangerrepresente_pays VARCHAR(100),
    informationsdeclarees_datedernieredeclaration DATE,
    informationsdeclarees_debutexercice DATE,
    informationsdeclarees_finexercice DATE,
    informationsdeclarees_specialitesdeformation_codespecialite1 INT,
    informationsdeclarees_specialitesdeformation_libellespecialite1 VARCHAR(255),
    informationsdeclarees_specialitesdeformation_codespecialite2 INT,
    informationsdeclarees_specialitesdeformation_libellespecialite2 VARCHAR(255),
    informationsdeclarees_specialitesdeformation_codespecialite3 INT,
    informationsdeclarees_specialitesdeformation_libellespecialite3 VARCHAR(255),
    informationsdeclarees_nbstagiaires INT,
    informationsdeclarees_nbstagiairesconfiesparunautreof INT,
    informationsdeclarees_effectifformateurs INT,
    com_arm_code INT,
    com_arm_name VARCHAR(255),
    epci_code INT,
    epci_name VARCHAR(255),
    dep_code INT,
    dep_name VARCHAR(100),
    reg_name VARCHAR(100),
    reg_code INT,
    toutes_specialites TEXT,
    organisme_formation_geocode BIT,
    certifications VARCHAR(255),
    random_id FLOAT,
    start_date DATE NOT NULL,
    end_date DATE,  
    is_active BIT NOT NULL DEFAULT 1,
    created_at DATE NOT NULL DEFAULT GETDATE(),
    updated_at DATE NOT NULL DEFAULT GETDATE(),
    CONSTRAINT UC_siretetablissementdeclarant_is_active UNIQUE (siretetablissementdeclarant, is_active)
    );
END
"""

# Function to fetch CSV data
def fetch_csv_data():
    try:
        logger.info("Fetching CSV data from API...")
        response = requests.get(API_URL, params={"delimiter": ";"
                                                 })
        response.raise_for_status()
        # Save CSV content to a temporary file
        with open('temp_data.csv', 'w', encoding='utf-8') as f:
            f.write(response.text)
        # Read CSV with pandas
        df = pd.read_csv('temp_data.csv', sep=';', low_memory=False)
        logger.info(f"Successfully fetched {len(df)} records.")
        return df
    except Exception as e:
        raise

# Function to clean data
def clean_data(df):
    try:
        logger.info("Cleaning data...")

        # Drop completely empty columns inplace
        df.dropna(axis=1, how='all', inplace=True)

        # Replace "Unknown" with NaN
        df.replace("Unknown", np.nan, inplace=True)

        # Convert date columns
        date_cols = [
            'informationsdeclarees_datedernieredeclaration',
            'informationsdeclarees_debutexercice',
            'informationsdeclarees_finexercice',
            'start_date',
            'end_date',
            'created_at',
            'updated_at'
        ]
        for col in date_cols:
            if col in df.columns:
                df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')

        # Convert integer columns (SQL: INT)
        int_cols = [
            "numerodeclarationactivite",
            "numerosdeclarationactiviteprecedent",
            "siren",
            "adressephysiqueorganismeformation_codepostal",
            "adressephysiqueorganismeformation_coderegion",
            "informationsdeclarees_specialitesdeformation_codespecialite1",
            "informationsdeclarees_specialitesdeformation_codespecialite2",
            "informationsdeclarees_specialitesdeformation_codespecialite3",
            "informationsdeclarees_nbstagiaires",
            "informationsdeclarees_nbstagiairesconfiesparunautreof",
            "informationsdeclarees_effectifformateurs",
            "com_arm_code",
            "epci_code",
            "dep_code",
            "reg_code"
        ]
        for col in ["numerodeclarationactivite", "numerosdeclarationactiviteprecedent","siretetablissementdeclarant"]:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Use Int64 for nullable integers
        # Handle other integer columns
        for col in int_cols:
            if col in df.columns and col not in ["numerodeclarationactivite", "numerosdeclarationactiviteprecedent"]:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        # Convert float columns (SQL: FLOAT)
        float_cols = [
            "random_id",
            "geocodageban"
        ]
        for col in float_cols:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

        # Convert string columns (SQL: VARCHAR/NVARCHAR)
        str_cols = [
              "denomination",
            "adressephysiqueorganismeformation_voie", "adressephysiqueorganismeformation_ville", 
            "organismeetrangerrepresente_denomination", "organismeetrangerrepresente_voie",
            "organismeetrangerrepresente_codepostal", "organismeetrangerrepresente_ville",
            "organismeetrangerrepresente_pays", "informationsdeclarees_specialitesdeformation_libellespecialite1",
            "informationsdeclarees_specialitesdeformation_libellespecialite2",
            "informationsdeclarees_specialitesdeformation_libellespecialite3", 
            "com_arm_name", "epci_name",  "dep_name", "reg_name", "toutes_specialites",
            "certifications"
        ]
        for col in str_cols:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(str).str.strip()

        # Convert boolean columns (SQL: BIT)
        bool_cols = [ "certifications_actionsdeformation", "certifications_bilansdecompetences",
            "certifications_vae", "certifications_actionsdeformationparapprentissage","organisme_formation_geocode", "is_active"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype('boolean')
                # Fill missing values with False
                df[col] = df[col].fillna(False)

        # Fill missing values
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].fillna("Unknown")
            elif pd.api.types.is_numeric_dtype(df[column]):
                df.loc[:, column] = df[column].fillna(0)
            elif df[column].dtype == 'boolean':
                df.loc[:, column] = df[column].fillna(False)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = df[column].fillna(pd.Timestamp("0001-01-01 00:00:00"))

        logger.info("Data cleaning completed.")
        return df
    except Exception as e:
        raise

# Function to connect to SQL Server and create table
def get_sql_connection():
    try:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD}"
            
        )
        conn = pyodbc.connect(conn_str)
        logger.info("Connected to SQL Server.")
        # Create table if it doesn't exist
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        logger.info("Table 'training_organizations' ensured.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQL Server or create table: {e}")
        raise

# Function to compare rows
def rows_are_different(existing_row, new_row, columns_to_compare):
    for col in columns_to_compare:
        existing_val = existing_row.get(col)
        new_val = new_row.get(col)

        # Handle NaN or None
        if pd.isna(existing_val) and pd.isna(new_val):
            continue

        # Handle float vs int: 0 == 0.0
        if isinstance(existing_val, (int, float)) and isinstance(new_val, (int, float)):
            if math.isclose(existing_val, new_val, rel_tol=1e-9):
                continue
            else:
                return True

        # Handle datetime.date vs pd.Timestamp
        if isinstance(existing_val, datetime.date) and isinstance(new_val, pd.Timestamp):
            if existing_val == new_val.date():
                continue
            else:
                return True

        # Compare normalized strings
        if isinstance(existing_val, str) or isinstance(new_val, str):
            if str(existing_val).strip() != str(new_val).strip():
                return True
            else:
                continue

        # Fallback comparison
        if existing_val != new_val:
            return True

    return False


# Function to process and insert data
def process_data():
    try:
        # Fetch data
        df = fetch_csv_data()

        # Clean data
        df = clean_data(df)

        # Prepare data for SQL
        df = df.where(pd.notnull(df), None)  # Replace NaN with None for SQL
        today = date.today()

        # Define columns to compare (exclude system-managed columns)
        columns_to_compare = [
            'numerodeclarationactivite', 'denomination', 'siren',
            'siretetablissementdeclarant', 'adressephysiqueorganismeformation_voie',
            'adressephysiqueorganismeformation_codepostal',
            'adressephysiqueorganismeformation_ville',
            'adressephysiqueorganismeformation_coderegion',
            'certifications_actionsdeformation', 'certifications_bilansdecompetences',
            'certifications_vae', 'certifications_actionsdeformationparapprentissage',
            'informationsdeclarees_datedernieredeclaration',
            'informationsdeclarees_nbstagiaires', 'informationsdeclarees_effectifformateurs'
        ]

        # Connect to SQL Server
        conn = get_sql_connection()
        cursor = conn.cursor()

        # Process each row
        for _, row in df.iterrows():
            try:
                siret = row['siretetablissementdeclarant']
                # Check if an active record exists for this SIRET
                cursor.execute("""
                    SELECT * FROM training_organizations
                    WHERE siretetablissementdeclarant = ? AND is_active = 1
                """, (siret,))
                existing_row = cursor.fetchone()

                if not existing_row:
                    # Insert new row
                    cursor.execute("""
                        INSERT INTO training_organizations (
                            numerodeclarationactivite, numerosdeclarationactiviteprecedent,
                            denomination, siren, siretetablissementdeclarant,
                            adressephysiqueorganismeformation_voie,
                            adressephysiqueorganismeformation_codepostal,
                            adressephysiqueorganismeformation_ville,
                            adressephysiqueorganismeformation_coderegion,
                            geocodageban, certifications_actionsdeformation,
                            certifications_bilansdecompetences, certifications_vae,
                            certifications_actionsdeformationparapprentissage,
                            organismeetrangerrepresente_denomination,
                            organismeetrangerrepresente_voie,
                            organismeetrangerrepresente_codepostal,
                            organismeetrangerrepresente_ville,
                            organismeetrangerrepresente_pays,
                            informationsdeclarees_datedernieredeclaration,
                            informationsdeclarees_debutexercice,
                            informationsdeclarees_finexercice,
                            informationsdeclarees_specialitesdeformation_codespecialite1,
                            informationsdeclarees_specialitesdeformation_libellespecialite1,
                            informationsdeclarees_specialitesdeformation_codespecialite2,
                            informationsdeclarees_specialitesdeformation_libellespecialite2,
                            informationsdeclarees_specialitesdeformation_codespecialite3,
                            informationsdeclarees_specialitesdeformation_libellespecialite3,
                            informationsdeclarees_nbstagiaires,
                            informationsdeclarees_nbstagiairesconfiesparunautreof,
                            informationsdeclarees_effectifformateurs,
                            com_arm_code, com_arm_name, epci_code, epci_name,
                            dep_code, dep_name, reg_name, reg_code,
                            toutes_specialites, organisme_formation_geocode,
                            certifications, random_id,
                            start_date, is_active, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, GETDATE(), GETDATE())
                    """, (
                        row.get('numerodeclarationactivite'), row.get('numerosdeclarationactiviteprecedent'),
                        row.get('denomination'), row.get('siren'), row.get('siretetablissementdeclarant'),
                        row.get('adressephysiqueorganismeformation_voie'),
                        row.get('adressephysiqueorganismeformation_codepostal'),
                        row.get('adressephysiqueorganismeformation_ville'),
                        row.get('adressephysiqueorganismeformation_coderegion'),
                        row.get('geocodageban'), row.get('certifications_actionsdeformation'),
                        row.get('certifications_bilansdecompetences'), row.get('certifications_vae'),
                        row.get('certifications_actionsdeformationparapprentissage'),
                        row.get('organismeetrangerrepresente_denomination'),
                        row.get('organismeetrangerrepresente_voie'),
                        row.get('organismeetrangerrepresente_codepostal'),
                        row.get('organismeetrangerrepresente_ville'),
                        row.get('organismeetrangerrepresente_pays'),
                        row.get('informationsdeclarees_datedernieredeclaration'),
                        row.get('informationsdeclarees_debutexercice'),
                        row.get('informationsdeclarees_finexercice'),
                        row.get('informationsdeclarees_specialitesdeformation_codespecialite1'),
                        row.get('informationsdeclarees_specialitesdeformation_libellespecialite1'),
                        row.get('informationsdeclarees_specialitesdeformation_codespecialite2'),
                        row.get('informationsdeclarees_specialitesdeformation_libellespecialite2'),
                        row.get('informationsdeclarees_specialitesdeformation_codespecialite3'),
                        row.get('informationsdeclarees_specialitesdeformation_libellespecialite3'),
                        row.get('informationsdeclarees_nbstagiaires'),
                        row.get('informationsdeclarees_nbstagiairesconfiesparunautreof'),
                        row.get('informationsdeclarees_effectifformateurs'),
                        row.get('com_arm_code'), row.get('com_arm_name'),
                        row.get('epci_code'), row.get('epci_name'),
                        row.get('dep_code'), row.get('dep_name'),
                        row.get('reg_name'), row.get('reg_code'),
                        row.get('toutes_specialites'), row.get('organisme_formation_geocode'),
                        row.get('certifications'), row.get('random_id'),
                        today
                    ))
                    logger.info(f"Inserted new record for SIRET: {siret}")
                else:
                    # Convert existing row to dict for comparison
                    existing_row_dict = {desc[0]: value for desc, value in zip(cursor.description, existing_row)}
                    new_row_dict = row.to_dict()
                    
                    if rows_are_different(existing_row_dict, new_row_dict, columns_to_compare):
                        # Mark existing row as closed
                        cursor.execute("""
                            UPDATE training_organizations
                            SET is_active = 0, end_date = ?, updated_at = GETDATE()
                            WHERE siretetablissementdeclarant = ? AND is_active = 1
                        """, (today, siret))

                        # Insert new row with same start_date
                        cursor.execute("""
                            INSERT INTO training_organizations (
                                numerodeclarationactivite, numerosdeclarationactiviteprecedent,
                                denomination, siren, siretetablissementdeclarant,
                                adressephysiqueorganismeformation_voie,
                                adressephysiqueorganismeformation_codepostal,
                                adressephysiqueorganismeformation_ville,
                                adressephysiqueorganismeformation_coderegion,
                                geocodageban, certifications_actionsdeformation,
                                certifications_bilansdecompetences, certifications_vae,
                                certifications_actionsdeformationparapprentissage,
                                organismeetrangerrepresente_denomination,
                                organismeetrangerrepresente_voie,
                                organismeetrangerrepresente_codepostal,
                                organismeetrangerrepresente_ville,
                                organismeetrangerrepresente_pays,
                                informationsdeclarees_datedernieredeclaration,
                                informationsdeclarees_debutexercice,
                                informationsdeclarees_finexercice,
                                informationsdeclarees_specialitesdeformation_codespecialite1,
                                informationsdeclarees_specialitesdeformation_libellespecialite1,
                                informationsdeclarees_specialitesdeformation_codespecialite2,
                                informationsdeclarees_specialitesdeformation_libellespecialite2,
                                informationsdeclarees_specialitesdeformation_codespecialite3,
                                informationsdeclarees_specialitesdeformation_libellespecialite3,
                                informationsdeclarees_nbstagiaires,
                                informationsdeclarees_nbstagiairesconfiesparunautreof,
                                informationsdeclarees_effectifformateurs,
                                com_arm_code, com_arm_name, epci_code, epci_name,
                                dep_code, dep_name, reg_name, reg_code,
                                toutes_specialites, organisme_formation_geocode,
                                certifications, random_id,
                                start_date, is_active, created_at, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, GETDATE(), GETDATE())
                        """, (
                            row.get('numerodeclarationactivite'), row.get('numerosdeclarationactiviteprecedent'),
                            row.get('denomination'), row.get('siren'), row.get('siretetablissementdeclarant'),
                            row.get('adressephysiqueorganismeformation_voie'),
                            row.get('adressephysiqueorganismeformation_codepostal'),
                            row.get('adressephysiqueorganismeformation_ville'),
                            row.get('adressephysiqueorganismeformation_coderegion'),
                            row.get('geocodageban'), row.get('certifications_actionsdeformation'),
                            row.get('certifications_bilansdecompetences'), row.get('certifications_vae'),
                            row.get('certifications_actionsdeformationparapprentissage'),
                            row.get('organismeetrangerrepresente_denomination'),
                            row.get('organismeetrangerrepresente_voie'),
                            row.get('organismeetrangerrepresente_codepostal'),
                            row.get('organismeetrangerrepresente_ville'),
                            row.get('organismeetrangerrepresente_pays'),
                            row.get('informationsdeclarees_datedernieredeclaration'),
                            row.get('informationsdeclarees_debutexercice'),
                            row.get('informationsdeclarees_finexercice'),
                            row.get('informationsdeclarees_specialitesdeformation_codespecialite1'),
                            row.get('informationsdeclarees_specialitesdeformation_libellespecialite1'),
                            row.get('informationsdeclarees_specialitesdeformation_codespecialite2'),
                            row.get('informationsdeclarees_specialitesdeformation_libellespecialite2'),
                            row.get('informationsdeclarees_specialitesdeformation_codespecialite3'),
                            row.get('informationsdeclarees_specialitesdeformation_libellespecialite3'),
                            row.get('informationsdeclarees_nbstagiaires'),
                            row.get('informationsdeclarees_nbstagiairesconfiesparunautreof'),
                            row.get('informationsdeclarees_effectifformateurs'),
                            row.get('com_arm_code'), row.get('com_arm_name'),
                            row.get('epci_code'), row.get('epci_name'),
                            row.get('dep_code'), row.get('dep_name'),
                            row.get('reg_name'), row.get('reg_code'),
                            row.get('toutes_specialites'), row.get('organisme_formation_geocode'),
                            row.get('certifications'), row.get('random_id'),
                            today
                        ))
                        logger.info(f"Updated record for SIRET: {siret}")
                    else:
                        logger.debug(f"No changes for SIRET: {siret}")

                conn.commit()

            except Exception as e:
                conn.rollback()
                continue

        cursor.close()
        conn.close()
        logger.info("Data processing completed.")

    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
# Function to schedule daily job
def run_daily_job():
    logger.info("Starting daily job...")
    process_data()

# Schedule the job to run daily at a specific time (e.g., 2 AM)
#schedule.every().day.at("02:00").do(run_daily_job)

# Main execution
if __name__ == "__main__":
    logger.info("Script started.")
    # Run once immediately for testing
    run_daily_job()
    # Keep the scheduler running
    #while True:
    #    schedule.run_pending()
     #   time.sleep(60)  # Check every minute