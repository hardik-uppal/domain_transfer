# Building the data


`python make_dataset.py`

positional arguments:

    jpeg_dir             path to jpeg images
    nb_channels          number of image channels

optional arguments:

    -h, --help           show this help message and exit
    --img_size IMG_SIZE  Desired Width == Height
    --do_plot            Plot the images to make sure the data processing went
                         OK



**Example:**

`python make_dataset.py pix2pix/datasets/dataset_name 3 --img_size 256 --do_plot True`