import os
import argparse


def launch_training(**kwargs):

    # Launch training
    train.train(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
#    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="CurtinFaces", help="CurtinFaces")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=25578, type=int, help="Number of batches per epoch")
    parser.add_argument('--nb_epoch', default=601, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch_classes', default=301, type=int, help="Number of training epochs for class specific model")
    parser.add_argument('--epoch', default=200, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--nb_classes', default=2, type=int, help="Number of classes")
    parser.add_argument('--do_plot', action="store_true", help="Debugging plot")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=256, type=int, help="Image width == height")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--logging_dir', default='../../curtin_feat_loss1', type=str, help="Path to logging directory")
    parser.add_argument('--load_weight_epoch', default=300, type=str, help="load weights from previous epochs")
    parser.add_argument('--batch_size_classes', default=4, type=float, help="batch size for class specific depth guided model")

    args = parser.parse_args()

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_data_format = "channels_first"
        K.set_image_data_format(image_data_format)
    elif args.backend == "tensorflow":
        image_data_format = "channels_last"
        K.set_image_data_format(image_data_format)

#    import train_3 as train
#    import train_wgan_2 as train
#    import train_wgan_pandora as train
    import train_gan_feat_loss as train        
#    import depth_classfier as train# train only depth images

    # Set default params
    d_params = {"dset": args.dset,
                "generator": args.generator,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "nb_epoch_classes": args.nb_epoch_classes,
                "model_name": "CNN",
                "epoch": args.epoch,
                "nb_classes": args.nb_classes,
                "do_plot": args.do_plot,
                "image_data_format": image_data_format,
                "bn_mode": args.bn_mode,
                "img_dim": args.img_dim,
                "use_label_smoothing": args.use_label_smoothing,
                "label_flipping": args.label_flipping,
                "patch_size": (64,64),
                "use_mbd": args.use_mbd,
                "logging_dir": args.logging_dir,
                "load_weight_epoch": args.load_weight_epoch,
                "batch_size_classes": args.batch_size_classes
                }

    # Launch training
    launch_training(**d_params)
