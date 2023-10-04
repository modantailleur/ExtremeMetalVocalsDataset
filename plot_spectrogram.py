import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
from torchaudio.transforms import MelSpectrogram
import torch

def plot_multi_spectro(x_m, z_m, fs, title_x, title_z, vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', cbar_label="Power (dB)", save=False, y_n_fft=None):
    if vmin is None:
        all_data = np.concatenate([x_m, z_m])
        vmin = np.min(all_data)
    if vmax is None:
        all_data = np.concatenate([x_m, z_m])
        vmax = np.max(all_data)

    exthmin = 1
    exthmax = len(x_m[0])
    extlmin = 0
    extlmax = 1

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20

    num_rows = 2  # Number of rows (you can adjust this as needed)
    num_x_spectrograms = len(x_m)
    num_z_spectrograms = len(z_m)
    max_spectrograms = max(num_x_spectrograms, num_z_spectrograms)

    fig, axs = plt.subplots(nrows=num_rows, ncols=max_spectrograms, sharey=True, figsize=(max_spectrograms * 5, num_rows * 4))

    for i, ax_row in enumerate(axs):
        print('AAAAAAAA')
        print(i)
        for j, ax in enumerate(ax_row):
            print('BBBBBBBBBBB')
            print(j)
            if i == 0:
                spectro_data = x_m[j]
                title = title_x[j]
            else:
                spectro_data = z_m[j]
                title = title_z[j]

            if j == 0:
                ylabel_ = ylabel
            else:
                ylabel_ = ''
            
            if diff:
                im = ax.imshow(spectro_data, extent=[extlmin, extlmax, exthmin, exthmax], cmap='seismic_r',
                               vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
            else:
                im = ax.imshow(spectro_data, extent=[extlmin, extlmax, exthmin, exthmax], cmap='inferno_r',
                               vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

            ax.set_title(title)
            ax.set_ylabel(ylabel_)

            if y_n_fft is not None:
                num_y_ticks = len(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks()*(fs/y_n_fft)/2)


    fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')

    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    axs[0, 0].set_ylabel(ylabel)
    fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)

    if save:
        plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.show()

# def plot_multi_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False):
#     if vmin==None:
#         vmin = np.min(x_m)
#     if vmax==None:
#         vmax = np.max(x_m)
#     exthmin = 1
#     exthmax = len(x_m[0])
#     extlmin = 0
#     extlmax = 1

#     mpl.rcParams['font.family'] = 'Times New Roman'
#     mpl.rcParams['font.size'] = 20
#     #mpl.use("pgf")
#     # mpl.rcParams.update({
#     #     "pgf.texsystem": "pdflatex",
#     #     'font.family': 'Times New Roman',
#     #     'text.usetex': True,
#     #     'pgf.rcfonts': False,
#     # })

#     #fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1]})
#     fig, axs = plt.subplots(ncols=len(x_m), sharey=True, figsize=(len(x_m)*5, 4))
#     #fig.subplots_adjust(wspace=1)

#     for i, ax in enumerate(axs):
#         if i == 0:
#             ylabel_ = ylabel
#         else:
#             ylabel_ = ''
#         if diff:
#             im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
#                     vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
#         else:
#             im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
#                     vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

#         ax.set_title(title[i])
#         #ax.set_xlabel('Time (s)')
#         ax.set_ylabel(ylabel_)

#     fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')
    
#     #cbar_ax = fig.add_axes([0.06, 0.15, 0.01, 0.7])
#     cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
#     cbar = fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
#     cbar.ax.yaxis.set_label_position('left')
#     cbar.ax.yaxis.set_ticks_position('left')

#     axs[0].set_ylabel(ylabel)
#     #fig.tight_layout()
#     #fig.tight_layout(rect=[0.1, 0.05, 1, 1], pad=2)
#     fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)
#     #fig.savefig('fig_spectro' + name + '.pdf', bbox_inches='tight', dpi=fig.dpi)
#     if save:
#         plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
#     plt.show()


n_frames = 64
n_fft=1024
hop_length = 256
n_mels = 128
fmin=20
fmax=8000
flen = 1024    
hlen = 256
window = 'hann'
norm = "slaney"
mel_scale = "slaney"

melspec_layer = MelSpectrogram(
    n_mels=n_mels,
    sample_rate=48000,
    n_fft=n_fft,
    win_length=flen,
    hop_length=hlen,
    f_min=fmin,
    f_max=fmax,
    center=True,
    power=2.0,
    mel_scale=mel_scale,
    norm=norm,
    normalized=True,
    pad_mode="constant",
)

audios_to_show = ['audio_examples/Singer2_ClearVoice_Mid_a.wav', 'audio_examples/Singer1_BlackShriek_Mid_a.wav', 'audio_examples/Singer13_DeathGrowl_Mid_a.wav', 'audio_examples/Singer15_HardcoreScream_Mid_a.wav']
x_mels_list = []

for f_name in audios_to_show:

    audio, sr = librosa.load(f_name, sr=48000)
    audio = librosa.util.normalize(audio)[48000:96000]
    spec = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)
    spec = np.abs(spec)
    # spec = np.log(spec+10e-10)

    # mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, window=window)
    # mels = np.log(mels + 10e-10) + 94
    # x_wave = torch.Tensor(audio).unsqueeze(0)
    # torch_mels = melspec_layer(x_wave)
    # torch_mels = 10 * torch.log10(torch_mels + 1e-10)
    # torch_mels = torch.clamp((torch_mels + 100) / 100, min=0.0)
    # mels = torch_mels.squeeze(0).numpy()
    x_mels_list.append(spec[:100, :])

# plot_multi_spectro(mels_list[:], 48000, title=['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream', 'Grind Inhale'], vmin=84, vmax=97)

audios_to_show = ['audio_examples/Singer12_GrindInhale_a.wav', 'audio_examples/Singer12_Effect_PigSqueal.wav', 'audio_examples/Singer14_Effect_DeepGutturals.wav', 'audio_examples/Singer10_Effect_TunnelThroat.wav']
z_mels_list = []

for f_name in audios_to_show:

    audio, sr = librosa.load(f_name, sr=48000)
    audio = librosa.util.normalize(audio)[48000:96000]
    spec = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)
    spec = np.abs(spec)
    # spec = np.log(spec+10e-10)
    # mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # mels = np.log(mels + 10e-10) + 94
    print(spec.shape)
    z_mels_list.append(spec[:100, :])

plot_multi_spectro(x_mels_list[:], z_mels_list[:], 48000, title_x=['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream'], title_z=['Grind Inhale', 'Pig Squeal', 'Deep Gutturals', 'Tunnel Throat'], ylabel='Frequency (Hz)', y_n_fft=512, vmin=0, vmax=15, cbar_label='Amplitude')
