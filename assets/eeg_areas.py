# This file contains the regions of interest (RoI) for EEG data analysis. 
# Each RoI is defined by a set of electrode indices and corresponding labels.

RoI_7 = {
    "Prefrontal": [53, 54, 55, 56, 28, 79, 80, 81],
    "Frontal": [9, 26, 27, 29, 30, 51, 52, 57, 77, 78, 82, 83, 103, 104, 105],
    "Central": [
        10, 25, 31, 50, 58, 11, 24, 32, 49, 59, 76, 84, 102, 106, 75, 85, 101, 107,
        6, 12, 23, 33, 48, 60, 74, 86, 100, 108, 114, 5, 13, 22, 34, 47, 73, 87, 99, 109, 113
    ],
    "Left-Temporal": [0, 1, 2, 3, 4, 7, 8],
    "Right-Temporal": [115, 116, 117, 118, 119, 120, 121],
    "Parietal": [
        14, 21, 35, 46, 61, 15, 20, 36, 45, 62, 72, 88, 98, 110, 71, 89, 97, 111
    ],
    "Occipital": [
        16, 17, 18, 19, 37, 38, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 112, 94, 95,
        96, 90, 91, 92, 93, 67, 68, 69, 70
    ]
}
RoI_Area7_labels = {
    "Prefrontal": ['AF3', 'FP1', 'FPz', 'AFz', 'AF7', 'AF4', 'Fp2', 'AF8'],
    "Frontal":  ['FFT7h', 'F5', 'F7', 'AFF5h', 'F3', 'FFC1h', 'F1', 'Fz', 'FFC2h', 'F2', 'AFF6h', 'F4', 'F6', 'F8', 'FFT8h'],
    "Central": ['FC5', 'FFC5h', 'FFC3h', 'FC1', 'FCz', 'FCC5h', 'FC3', 'FCC3h', 'FCC1h', 'FCCz', 'FC2', 'FFC4h', 'FFC6h', 'FC6', 'FCC2h', 'FCC4h', 'FC4', 'FCC6h',
                'C5', 'CCP5h', 'C3', 'C1', 'CCP1h', 'Cz', 'CCP2h', 'C2', 'C4', 'CCP6h', 'C6','TTP7h', 'TPP5h', 'CP3', 'CCP3h', 'CP1', 'CP2', 'CCP4h', 'CP4', 'TPP8h', 'TTP8h'],
    "Left-Temporal": ["T9", "FT9", "FTT9H", "T7", "TP7", "FTT7H", "FT7"],
    "Right-Temporal": ["FTT8H", "FT8", "T10", "FT10", "FTT10H", "T8", "TP8"],
    "Parietal": ["P7", "CPP5H", "CPP3H", "CPP1H", "CPPZ", "P9", "P5", "P3", "PPO1", "PPOZ", "CPP2H", "CPP4H", "CPP6H", "P8", "PPO2", "P4", "P6", "P10"],
    "Occipital": ["P11", "PO11", "PO9", "PPO7", "PO3", "POO7", "POO9H", "POO11H", "I1", "OI1", "POO3", "PO1", "POZ", "POOZ", "OZ", "IZ", "P12", "PO12", "PO10",
                  "PPO8", "PO4", "POO8", "POO10H", "POO12H", "I2", "OI2", "POO4", "PO2"]
}



RoI_5 = {
    "Frontal": [53, 54, 55, 56, 28, 79, 80, 81, 9, 26, 27, 29, 30, 51, 52, 57, 77, 78, 82, 83, 103, 104, 105],
    "Central": [
        10, 25, 31, 50, 58, 11, 24, 32, 49, 59, 76, 84, 102, 106, 75, 85, 101, 107,
        6, 12, 23, 33, 48, 60, 74, 86, 100, 108, 114, 5, 13, 22, 34, 47, 73, 87, 99, 109, 113],
    "Left-Temporal": [0, 1, 2, 3, 4, 7, 8, 115, 116, 117, 118, 119, 120, 121],
    "Parietal": [14, 21, 35, 46, 61, 15, 20, 36, 45, 62, 72, 88, 98, 110, 71, 89, 97, 111],
    "Occipital": [
        16, 17, 18, 19, 37, 38, 39, 40, 41, 42, 43, 44, 63, 64, 65, 66, 112, 94, 95,
        96, 90, 91, 92, 93, 67, 68, 69, 70
    ]    
}


region_to_labels = {
    "Frontal": [
        "AF7", "FP1", "FPZ", "AFZ", "AFF5H", "AF4", "FP2", "AF8", 
        "FFT7H", "F5", "F7", "AFF5H", "F3", "FFC1H", "F1", "FZ", 
        "FFC2H", "F2", "AFF6H", "F4", "FFC4H", "F8", "FFT8H"
    ],
    "Central": [
        "FC5", "FFC5H", "FFC3H", "FC1", "FCZ", "FCC5H", "FC3", "FCC3H", "FCC1H", 
        "FCCZ", "FC2", "FFC4H", "FFC6H", "FC6", "FCC2H", "FCC4H", "FC4", "FCC6H",
        "C5", "CCP5H", "C3", "C1", "CCP1H", "CZ", "CCP2H", "C2", "C4", "CCP4H", "C6",
        "TTP7H", "TPP5H", "CP3", "CCP3H", "CP1", "CP2", "CCP4H", "CP4", "TPP8H", "TTP8H"
    ],
    "Left-Temporal": [
        "T9", "FT9", "FTT9H", "T7", "TP7", "FTT7H", "FT7", 
        "FTT8H", "FT8", "T10", "FT10", "FTT10H", "T8", "TP8"
    ],
    "Parietal": [
        "P7", "CPP5H", "CPP3H", "CPP1H", "CPPZ", "P9", "P5", 
        "P3", "PPO1", "PPOZ", "CPP2H", "CPP4H", "CPP6H", "P8", 
        "PPO2", "P4", "P6", "P10"
    ],
    "Occipital": [
        "P11", "PO11", "PO9", "PPO7", "PO3", "POO7", "POO9H", "POO11H", 
        "I1", "OI1", "POO3", "PO1", "POZ", "POOZ", "OZ", "IZ", 
        "P12", "PO12", "PO10", "PPO8", "PO4", "POO8", "POO10H", "POO12H", 
        "I2", "OI2", "POO4", "PO2"
    ]
}

