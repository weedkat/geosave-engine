# This is a template for data ingestion that pipelines
# Modify this to implement auto fetch from new datasets
import questionary

def ingest(bbox=None, time_range=None, **kwargs):
    # This function is responsible for ingesting raw data from various sources and saving it in a standardized format for preprocessing.
    # You can implement logic to fetch data from APIs, read from files, or any other data source relevant to your use case.
    pass  # No-op, data ingestion logic is not implemented in this template

if __name__ == "__main__":
    ingest()
