EPOCHS = 1000
PRIVATE_DATA_PATH = './data/private'


r = 5 # Reduction factor. Paper => 2, 3, 5
sr = 22050 # sample rate
preemphasis = .97 # or None

n_fft = 2048 # fft points (samples)
n_mels = 80 # Number of Mel banks to generate
n_iter = 50 # Number of inversion iterations

max_db = 100
ref_db = 20

frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds

hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
