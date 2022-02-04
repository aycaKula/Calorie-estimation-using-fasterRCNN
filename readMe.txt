I have worked inside the Tensorflow Object Detection API. 
And I have added my final_code inside the object_detection 
folder (which is a folder inside the API).

I have included the segmentation, and calculations part that are
specific for my project.

run_inference --> inside this function I have applied the grab-cut method.
get_volume --> volume calculation and fill the csv


Commands for training and testing the data in tensorflow API
train:
python model_main_tf2.py --pipeline_config_path=training/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.config --model_dir=training --alsologtostderr  --sample_1_of_n_eval_examples=1 --num_eval_steps=1000
create model after train:
python exporter_main_v2.py --trained_checkpoint_dir=training --pipeline_config_path=training/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.config --output_directory inference_graph
test model:
python testModel.py --model=inference_graph\saved_model --labelmap=training\label_map.pbtxt --image_path=images/test/mix014S(3).JPG
python kingAycaTheEnd.py --model=inference_graph\saved_model --labelmap=training\label_map.pbtxt
mix003S(1).JPG
