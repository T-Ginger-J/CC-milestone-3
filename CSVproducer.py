from google.cloud import pubsub_v1      # pip install google-cloud-pubsub  ##to install
import glob                             # for searching for json file 
import json
import os 
import random
import numpy as np                      # pip install numpy    ##to install
import time

from csv import DictReader

# Search the current directory for the JSON file (including the service account key) 
# to set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

# Set the project_id with your project ID
project_id="milestone3-449602";
topic_name = "CSVrecords";   # change it for your topic name if needed

# create a publisher and get the topic path for the publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)
print(f"Published messages with ordering keys to {topic_path}.")

#read from csv into dictionary
with open("Labels.csv", 'r') as f:
    csvFile = DictReader(f)
    #read each row of the Dictionary
    for row in csvFile:
        msg = row
        print(msg)
        record_value=json.dumps(msg).encode('utf-8');    # serialize the message
        try:    
            future = publisher.publish(topic_path, record_value);
            #ensure that the publishing has been completed successfully
            future.result()    
            print("The messages {} has been published successfully".format(msg))
        except: 
            print("Failed to publish the message")
    
        
      