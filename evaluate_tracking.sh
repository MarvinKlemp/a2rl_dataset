# you need to fix the splits in /home/marvin/dev/dataset/venv/lib/python3.10/site-packages/nuscenes/utils/splits.py
# pip install motmetrics==1.1.3

python3 evaluate_tracking.py
python3 -m nuscenes.eval.tracking.evaluate --dataroot /home/marvin/dev/dataset/data/v1.0-mini --version v1.0-mini  --eval_set mini_train /home/marvin/dev/dataset/prediction_tracks_mini_train.json