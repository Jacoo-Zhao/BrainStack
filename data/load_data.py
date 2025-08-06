from data.dataset import SpeechWordDatasetV5

def load_data(subject_list, resample_fs, split_method='leave-one-session-out'):
    phase_list = ['train', 'val','test']
    dataset = {}  
    valid_session = [11]
    test_session = [12]
    if split_method == 'cross-subject':
        val_subject, test_subject = subject_list[:2]
    else:
        val_subject = test_subject = subject_list[0]
        
    for phase in phase_list:
        preproc_args_alpha = {
            "feature_type": "wave",
            "format": "curry",
            "output_format": "epoch",
            "pp_postfix": "_cleaned_lo_1hi_75_noref_.set",
            "wavelet_method": None,
            "fmin": 1,
            "fmax": 50,
            "fnum": None,
            "fspacing": None,
            "fspecial": None,
            "tmin": 0,
            "tmax": 1.0,
            "avg_ref": False,
            "resample_fs": resample_fs,
            "resample_freq_time": None,
            "n_jobs": 16,
            "batch_size": 128,
            "epoch_data": False,
            "cache_session": False,
            'relevant_events': ['100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123'],
        }
        
    
        dataset_alpha = SpeechWordDatasetV5(
            phase=phase, 
            data_root = 'data_root_pth',
            split_method=split_method, 
            subjects=subject_list,
            val_subject=val_subject,
            val_sessions=valid_session,
            test_subject=test_subject,
            test_sessions=test_session,
            select_channels=None,
            select_vocabs=None, 
            preproc_args=preproc_args_alpha,
            force_rebuild_index=True,
            force_rebuild_feature=False,
            debug=False,
            output_dict=True,
            random_seed=42,
            output_meta_label=True,
        )
        dataset[phase] = dataset_alpha
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']

    print(train_dataset[0].keys())
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, valid_session, test_session
