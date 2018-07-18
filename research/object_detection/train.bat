set PYTHONPATH=C:\cuda\test\models;C:\cuda\test\models\research;C:\cuda\test\models\research\slim
set CUDA_VISIBLE_DEVICES=0
python xml_to_csv.py
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

rem python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-200000 --output_directory inference_graph
rem python eval.py --logtostderr --checkpoint_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config --eval_dir=training/
