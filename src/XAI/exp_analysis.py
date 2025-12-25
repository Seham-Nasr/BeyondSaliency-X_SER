from Models.trainSER import ResNetSmall
from Models.SER_data import *
from XAI.xAIutils import *
from XAI.supplies import *
from XAI.features import extract_egmaps, plot_feature_bars, Validation
import torch

def explanation_analysis(data_setname, data_path, XAI_method, Emotion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #_____________________________dataset 1____________________________________
    if data_setname == "Crema-D":
        data_path = data_path # Path to the audio files for Crema-D
        LABEL_DICT = {0:'fear', 1:'neutral', 2:'happy',3:'angry', 4:'disgust', 5:'surprise',6:'sad'}
        LABELS = list(LABEL_DICT.values())
        #process_dataset(data_path=data_path,data_setname=data_setname)
        ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, LABELS =  process_dataset(data_path= data_path,
                                                                            data_setname='Crema-D',
                                                                            pickle_file='../src/Models/data/Crema_D.pkl',
                                                                            label_column='Emotions'
                                                                            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model
        classes = 6
        model = ResNetSmall(classes)
        checkpoint_path = "../src/Models/checkpoint/best_model_Crema.pth"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Map emotions to selected instance indices from the test dataset
        # Note: The indices below are based on the dataset's structure and may need adjustment.
        '''instance_map = {
            "sad": 140,
            "neutral": 723,
            "angry": 329,
            "happy": 721,
            "fear": 39,
            "sad": 5,
            "neutral": 12,
            "angry": 6,
            "happy": 1,
            "fear": 42

        }'''

        #True prediction instances
        instance_map = {
            "sad": 140,
            "neutral": 615,
            "angry": 329,
            "happy": 672,
            "fear": 627
        }
        # False prediction instances
        '''instance_map = {
            "sad": 0,
            "neutral": 1,
            "angry": 3,
            "happy": 6,
            "fear": 33
        }'''

        if Emotion not in instance_map:
            raise ValueError("Invalid emotion selected. Choose from 'happy', 'sad', 'angry', 'neutral', or 'fear'.")

        index = instance_map[Emotion]
        ima = ds_test.data[index][1]
        label = ds_test.labels[index]
        data = ds_test.data[index][0]
        
        print(data)
        # Prediction
        pred = get_prediction(model, data=data)
        print(f"Predicted label: {LABEL_DICT[pred]}")
        print(f"True Label: {label} (Maps to: {LABEL_DICT[label]})")

        # XAI
        if XAI_method == "Occlusion":
            windows = Occlusion_xai(model, data, LABEL_DICT, data_setname, LABEL_DICT[label], device)
            print("eGMAPS feature extraction in progress...")
            print("Time steps...", windows)
            # Extract eGMAPS features
            lld1 = extract_egmaps(data, windows, sr=16000)
            # plot selected acoustic features
            stats = plot_feature_bars(lld1, row=0, aggregate=True, top_n=15)
            print("\n______________\n",stats)
        elif XAI_method == "CRP":
            windows = CRP_xai(model, data, ima, LABEL_DICT, data_setname,LABEL_DICT[label], device)

            print("eGMAPS feature extraction in progress...")
            print("Time steps...", windows)
            # Extract eGMAPS features
            lld1 = extract_egmaps(data, windows, sr=16000)
            # plot selected acoustic features
            stats = plot_feature_bars(lld1, row=0, aggregate=True, top_n=15)
            print("\n___________________________\n",stats,"\n_______________________\n")
            lld2 = Validation(data)
                


        else:
            raise ValueError("Invalid XAI method selected. Choose 'GradCAM' or 'Occlusion'.")
        
 #____________________________________dataset: 2_________________________
    elif data_setname == "TESS":
        data_path = "../src/Models/TESS/tess/tess/"# Path to the audio files for Crema-D
        LABEL_DICT = {0:'fear', 1:'neutral', 2:'happy',3:'angry', 4:'disgust', 5:'surprise',6:'sad'}
        LABELS = list(LABEL_DICT.values())
        ds_train, ds_test, ds_val, dl_train, dl_test, dl_val, LABELS = process_dataset(
                                                                        data_path='../src/Models/data/TESS/tess/tess/',
                                                                        data_setname='TESS',
                                                                        pickle_file='../src/Models/data/TESS_df.pkl',
                                                                        label_column='Emotions'
                                                                    )
        
        # Load the model
        classes = 7
        model = ResNetSmall(classes)
        checkpoint_path = "../src/Models/checkpoint/best_model_tess.pth"
        model = ResNetSmall(len(LABELS))
        model.load_state_dict(torch.load(checkpoint_path ))
        model.to('cuda')
        model.eval()

        # Emotion and instance selection    
        # Map emotions to selected instance indices from the test dataset
        # True prediction instances
        '''instance_map = {
            "sad": 40,
            "neutral": 68,
            "angry": 17,
            "happy": 96,
            "fear": 7
            
        }'''
        # False prediction instances
        instance_map = {
            "neutral": 68,
            "angry": 2,
            "happy": 3,
            "fear": 4,
            "sad": 0,
        }
        if Emotion not in instance_map:
            raise ValueError("Invalid emotion selected. Choose from 'happy', 'sad', 'angry', 'neutral', or 'fear'.")


        index = instance_map[Emotion]  # here to change  
        ima = ds_test.data[index][1]
        label = ds_test.labels[index]
        data = ds_test.data[index][0]
        pred = get_prediction(model, data = data)
        print(f"Predicted label: {LABEL_DICT[pred]}")
        print("True Label: ",label, "Maps to: ", LABEL_DICT[label])
        #data = "../" + data
        # XAI 
        if XAI_method == "Occlusion":
                windows = Occlusion_xai(model, data, LABEL_DICT, data_setname,LABEL_DICT[label], device)
                
                print("eGMAPS feature extraction in progress...")
                print("Time steps...", windows)
                # Extract eGMAPS features
                lld1 = extract_egmaps(data, windows, sr=16000)
                # plot selected acoustic features
                stats = plot_feature_bars(lld1, row=0, aggregate=True, top_n=15)
                print("\n______________\n",stats)
        elif XAI_method == "CRP":
                windows = CRP_xai(model, data, ima, LABEL_DICT, data_setname,LABEL_DICT[label], device)
                
                print("eGMAPS feature extraction in progress...")
                print("Time steps...", windows)
                # Extract eGMAPS features
                lld1 = extract_egmaps(data, windows, sr=16000)
                # plot selected acoustic features
                stats = plot_feature_bars(lld1, row=0, aggregate=True, top_n=15)
                print("\n______________\n",stats)
        else:
                raise ValueError("Invalid XAI method selected. Choose either 'CRP' or 'Occlusion'.")
    else:
        raise ValueError("Invalid dataset name. Choose from 'Crema-D', 'SAVEE', 'TESS', or 'RAVDESS'.")
           
