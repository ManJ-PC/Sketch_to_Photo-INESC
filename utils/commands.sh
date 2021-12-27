# Unzip the dataset
unzip dataset.zip

# Resize source sketches
python tools/process.py \
  --input_dir dataset/sketches \
  --operation resize \
  --output_dir dataset/sketches_resized

# Resize source photos
python tools/process.py \
  --input_dir dataset/photos \
  --operation resize \
  --output_dir dataset/photos_resized

# Combine resized sketches with photos
python tools/process.py \
  --input_dir dataset/sketches_resized \
  --b_dir dataset/photos_resized \
  --operation combine \
  --output_dir dataset/combined  

# Split into train/val set
python tools/split.py \
  --dir dataset/combined

# train the model sketch to photo
python pix2pix.py \
  --mode train \
  --output_dir photos_train \
  --max_epochs 1000 \
  --input_dir dataset/combined/train/ \
  --which_direction AtoB

# train the model photo to sketch
python pix2pix.py \
  --mode train \
  --output_dir photos_train_B \
  --max_epochs 1000 \
  --input_dir dataset/combined/train/ \
  --which_direction BtoA

# test the model
python pix2pix.py \
      --mode test \
      --output_dir photos_test \
      --input_dir dataset/combined/val \
      --checkpoint photos_train
