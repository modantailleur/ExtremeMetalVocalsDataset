import argparse
import experiment_manager as em

def main(config):
    if config.dataset == 'kfold':  
        dataset_name = 'mel_dataset_kfold'
        distorsion_in_train = True

    if config.dataset == 'kfold_no_distorsion': 
        dataset_name = 'mel_dataset_kfold_no_distorsion'
        distorsion_in_train = False

    em.create_mel_dataset(dataset_name=dataset_name, distorsion_in_train=distorsion_in_train)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', help='experiment among the list: "kfold", "kfold_no_distorsion"')
    config = parser.parse_args()
    main(config)