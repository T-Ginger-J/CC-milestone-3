# environment variable setup for private key file
import os
import pandas as pd   #pip install pandas  ##to install

from google.cloud import pubsub_v1    #pip install google-cloud-pubsub  ##to install
import time
import json;
import io;
import glob

# Search the current directory for the JSON file (including the service account key) 
# to set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

# TODO : fill project id 
project_id = "milestone3-449602"
consuming_id = "CSVrecords"
subscription_id = "CSVrecords-sub"
producing_id = "converted"

# create a subscriber to the subscriber for the project using the subscription_id
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
topic_path = 'projects/{}/topics/{}'.format(project_id,consuming_id);

publisher = pubsub_v1.PublisherClient()
# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_id}`
producing_path = publisher.topic_path(project_id, producing_id)

#consume message
#process message
#produce message

for row in df.iterrows():
    value=row[1].to_dict()
    future = publisher.publish(producing_path, json.dumps(value).encode('utf-8'));
    print("Image with key "+str(value["ID"])+" is sent")
    time.sleep(0.1);
    
