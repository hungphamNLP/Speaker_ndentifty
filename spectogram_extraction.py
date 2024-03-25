import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os, errno
from pathlib import Path
import uuid
import python_speech_features as psf
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import random
import torchvision.datasets as dset


def preprocessing_adduser(audio,username):
    my_dpi = 120
    ay, sr = librosa.load(audio)
    duration = ay.shape[0] / sr
    print(duration)
    start = 0
    if os.path.exists('data_dir/'+username):
        print('username exist')
    else:
        os.makedirs('data_dir/'+username)
    while start + 5 < duration:
        slice_ = ay[start * sr: (start + 5) * sr]
        start = start + 5 - 1
        x = librosa.stft(slice_)
        xdb = librosa.amplitude_to_db(abs(x))
        plt.figure(figsize=(227 / my_dpi, 227 / my_dpi), dpi=my_dpi)
        plt.axis('off')
        librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='log')
        plt.savefig('data_dir/'+username+'/'+str(uuid.uuid4().hex + '.png'), dpi=my_dpi)
        

def processing_img(username,transform =transforms.Compose([transforms.ToTensor()])):
    folder_dataset = dset.ImageFolder(root='data_dir')
    class_index = folder_dataset.class_to_idx[username]
    
    class2_data = [data for data in folder_dataset.imgs if data[1] == class_index]
    print(len(class2_data))
    img = random.choice(class2_data)
    #we need to make sure approx 50% of images are in the same class
    should_get_same_class = 1
    if should_get_same_class:
        while True:
            img1_tuple = random.choice(class2_data) 
            if img[1]==img1_tuple[1]:
                break
    else:
        while True:
            img1_tuple = random.choice(class2_data) 
            if img[1] !=img1_tuple[1]:
                break
    
    img = Image.open(img[0])
    img = img.convert("L")  # conversion to gray
    img=PIL.ImageOps.invert(img)
    img = transform(img)
    print(img.shape)
    return img

    

def preprocessing_audio(folder_spec,root= 'datasets/LibriSpeech',audios='audio',type_audio='flac',my_dpi=120):## audio is dataset-train , audio-test is dataset-test
    speakers = pd.read_csv(root + '/SPEAKERS.TXT', sep='|',skipinitialspace=True,skip_blank_lines=True)
    speakers_filtered = speakers[(speakers['SUBSET'] == audios) | (speakers['SUBSET'] == audios)]
    speakers_filtered = speakers_filtered.copy()
    speakers_filtered['CODE'] = speakers_filtered['NAME'].astype('category').cat.codes
    print('read and filtered metdata')
    unique_speakers = np.unique(speakers_filtered['CODE'])
    for speaker in unique_speakers:
        try:
            os.makedirs(root + '/'+folder_spec+'/' + str(speaker))
            print('created folder for speaker {}'.format(speaker))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print('created all directories')  
    for index, row in speakers_filtered.iterrows():
        dir_ = root + '/' + row['SUBSET'] + '/' + str(row['ID']) + '/'
        print('working on df row {}, spaker {}'.format(index, row['CODE']))
        if not os.path.exists(dir_):
            print('dir {} not exists, skipping'.format(dir_))
            continue

        files_iter = Path(dir_).glob('**/*.'+type_audio)
        files_ = [str(f) for f in files_iter]

        for f in files_:
            ay, sr = librosa.load(f)
            duration = ay.shape[0] / sr
            start = 0
            while start + 5 < duration:
                slice_ = ay[start * sr: (start + 5) * sr]
                start = start + 5 - 1
                x = librosa.stft(slice_)
                xdb = librosa.amplitude_to_db(abs(x))
                plt.figure(figsize=(227 / my_dpi, 227 / my_dpi), dpi=my_dpi)
                plt.axis('off')
                librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='log')
                plt.savefig(root + '/'+folder_spec+'/' + str(row['CODE']) + '/' + uuid.uuid4().hex + '.png', dpi=my_dpi)
                plt.close()

        print('work done on index {}, speaker {}'.format(index, row['CODE']))


def get_fbanks(audio_file):
    
    def normalize_frames(signal, epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

    y, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        # if less than 1 seconds, don't use that audio
        return None

    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 15 ms. This will have an overlap of 10s
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    filter_banks = normalize_frames(signal=filter_banks)

    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks


def extract_fbanks(path):
    fbanks = get_fbanks(path)
    num_frames = fbanks.shape[0]

    # sample sets of 64 frames each

    numpy_arrays = []
    start = 0
    while start < num_frames + 64:
        slice_ = fbanks[start:start + 64]
        if slice_ is not None and slice_.shape[0] == 64:
            assert slice_.shape[0] == 64
            assert slice_.shape[1] == 64
            assert slice_.shape[2] == 1

            slice_ = np.moveaxis(slice_, 2, 0)
            slice_ = slice_.reshape((1, 1, 64, 64))
            numpy_arrays.append(slice_)
        start = start + 64

    print('num samples extracted: {}'.format(len(numpy_arrays)))
    return np.concatenate(numpy_arrays, axis=0)



if __name__ == '__main__':
    # preprocessing_audio(folder_spec='test-gram',audios='audio-test')
    # preprocessing_adduser('sample-1.wav','hungpham23')
    processing_img('hungpham23')