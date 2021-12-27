from subprocess import call

call([
    "python", "pix2pix/pix2pix.py", "--mode", "test", "--output_dir",
    "pix2pix/test", "--input_dir", "Images/combined", "--checkpoint",
    "pix2pix/photos_train"
])

print(13)
