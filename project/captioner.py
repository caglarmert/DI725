from inference import batch_pg_inference

model_path="PALIGEMMA_WEIGHTS"
prompt="PROMPT"
image_path="IMAGE_PATH"
output_csv_path="OUTPUT_PATH"
max_tokens=MAX_TOKENS
temp=TEMP
top_p=TOP_P
do_sample=DO_SAMPLE
batch_size = BATCH_SIZE

batch_pg_inference(model_path,image_path,output_csv_path,prompt,max_tokens,temp,top_p,do_sample,batch_size)