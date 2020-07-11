# TODO: convert to automated test
# example: writes frame.png → gs://kubric/subfolder/target.png
from google.cloud import storage
bucket = storage.Client().bucket("kubric") #< gs://kubric
blob = bucket.blob("subfolder/target.png")  #< position on bucket
blob.upload_from_filename("frame.png") #< position on local system