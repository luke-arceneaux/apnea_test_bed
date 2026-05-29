import torch
import argparse 

'''
A class to parse configs from the command line

You can instantiate this class like so: 
cfg, _ = DefaultConfig().parse()

'''
class DefaultConfig():

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # Paths/Files
        self.parser.add_argument("--data_dir",              default="../raw_data/",          help="raw input data")
        self.parser.add_argument("--processed_dir",         default="../processed_data/",     help="data for ML")
        self.parser.add_argument("--model_dir",             default="../models/",         help="path to save model")
        self.parser.add_argument("--plot_dir",             default="plots/",         help="path to save plots")
        self.parser.add_argument("--expert_annotations_dir",     default="../expert_annotations/",help="path to store expert annotations")
        self.parser.add_argument("--nst_annotations_dir",        default="nst_annot/",help="path to store NST-extracted annotations")
        self.parser.add_argument("--report_dir",             default="reports/",         help="path to save reports")
        self.parser.add_argument("--session_dir",             default="sessions/",         help="path to get session information")

        # Run info
        self.parser.add_argument("--dataset",              default="dreams",   help="dataset (dreams, dublin, mit, patch..)")
        self.parser.add_argument("--signal",              default="cannula",          help="signal (micfft, cannula, etc...)")
        self.parser.add_argument("--include_apnea",      default=True,      help="include apnea in list of apnea types to consider")
        self.parser.add_argument("--include_hypopnea",      default=False,      help="include hypopnea in list of apnea types to consider")
        self.parser.add_argument("--subject",              default=1,          help="subject/patient ID to use")
        self.parser.add_argument("--session",              default=1,          help="session # for subject")

        # Preprocessing
        self.parser.add_argument("--filter",            default='',      help="specify filter type (or None)")
        self.parser.add_argument("--normalize",            default=True,      help="normalize", action='store_true')

        # min num events to be extracted
        self.parser.add_argument("--threshold", default=0.1,          help="variance threshold for binarizing signal when detecting flatlines")
        self.parser.add_argument("--variance_threshold", default=0.1,          help="apnea variance threshold for binarizing signal when detecting flatlines")
        self.parser.add_argument("--hypopnea_variance_threshold", default=0.3,          help="hypopnea variance threshold for binarizing signal when detecting flatlines")

        self.parser.add_argument("--seconds_before_apnea",default=10,         help="# seconds before start of flatline to count as onset")
        self.parser.add_argument("--seconds_after_apnea", default=5,          help="# after start of flatline to count as onset")
        self.parser.add_argument("--min_apnea_seconds", default=10,          help="minimum seconds of flatline to count as apnea event")

        # nonlinear scaling
        self.parser.add_argument("--use_nonlinear_scaling",  default=False, help="whether or not to use nonlinear scaling in conditioning step")
        self.parser.add_argument("--slope_threshold", default=0.1,   help="slope threshold for nonlinear scaling")
        self.parser.add_argument("--scale_factor_low", default=0.5,  help="scale factor high for nonlinear scaling")
        self.parser.add_argument("--scale_factor_high", default=1,   help="scale factor low for nonlinear scaling")

   
        # Plot
        self.parser.add_argument("--plot_locally",      default=False,       help="plot manually annotated events")


        # Training/Testing
        self.parser.add_argument("--test_only",             default=False,      help="skips training and only performs inference", action='store_true')
        self.parser.add_argument("--test_frac",            default=0.2,        help="ratio of dataset to hold out for testing")
        self.parser.add_argument("--model_type",           default="cnn",      help="model type")
        self.parser.add_argument("--retrain",             default=False,      help="retrain", action='store_true')
        self.parser.add_argument("--learning_rate",       default=0.001,      help="learning rate")
        self.parser.add_argument("--batch_size",            default=16,         help="batch size")    
        self.parser.add_argument("--epochs",               default=5,         help="number of epochs to train")
        self.parser.add_argument("--save_model",            default=False,              help="save model", action='store_true')


    def parse(self):
        # parse args 
        return self.parser.parse_known_args()
