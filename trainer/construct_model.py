from model import BrainStack

def construct_eeg_model(
    option="brainstack",
    EEG_AREAS=None,
    num_classes=24,
    channels=122,
    dropout_rate=0.6,
    input_window_samples=1000,
    num_areas=7,
):
    print(f"Constructing model: {option}")
    if option == "brainStack":
        model = BrainStack(
            eeg_areas=EEG_AREAS,
            num_areas=num_areas,
            n_outputs=num_classes,
            n_chans=channels,
            att_heads=4,
            att_depth=2,
            n_times = input_window_samples,
            final_fc_length="auto",
            return_features=True
        )

    return model
