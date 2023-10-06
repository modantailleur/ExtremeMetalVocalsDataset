import argparse
import experiment_manager as em

def main(config):
    if config.exp == 't_multiclass':
        dataset_name = 'mel_dataset_kfold'
        out_name = 't_multiclass'
        model_prefix = 'model_t_multiclass'
        groundtruth='technique'

    if config.exp == 't_binary':
        dataset_name = 'mel_dataset_kfold'
        out_name = 't_binary'
        model_prefix = 'model_t_binary'
        groundtruth='technique_binary'

    if config.exp == 's_classif':  
        dataset_name = 'mel_dataset_kfold'
        out_name = 's_classif'
        model_prefix = 'model_s_classif'
        groundtruth='singer'

    if config.exp == 's_classif_no_distorsion': 
        dataset_name = 'mel_dataset_kfold_no_distorsion'
        out_name = 's_classif_no_distorsion'
        model_prefix = 'model_s_classif_no_distorsion'
        groundtruth='singer'
    
    if config.step == 'train':
        em.train_model(dataset_name=dataset_name, model_prefix=model_prefix, groundtruth=groundtruth)

    if config.step == 'eval':
        em.eval_model(dataset_name=dataset_name, model_prefix=model_prefix, groundtruth=groundtruth, out_name=out_name)

    if config.step == 'metric':
        em.calculate_metric(out_name=out_name, exp_type=groundtruth)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp', help='experiment among the list: "t_multiclass", "t_binary", "s_classif", "s_classif_no_distorsion"')
    parser.add_argument('-step', '--step', help='step among the list: "train", "eval", "metric"')
    config = parser.parse_args()
    main(config)