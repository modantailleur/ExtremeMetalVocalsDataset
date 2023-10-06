import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa

N_FRAMES = 64
N_FFT = 1024
HOP_LENGTH = 256
WINDOW = 'hann'
SR = 48000
F_MAX = 4700
FIRST_ROW_AUDIOS = ['audio_examples/Singer2_ClearVoice_Mid_a.wav', 'audio_examples/Singer1_BlackShriek_Mid_a.wav', 'audio_examples/Singer13_DeathGrowl_Mid_a.wav', 'audio_examples/Singer15_HardcoreScream_Mid_a.wav']
SECOND_ROW_AUDIOS = ['audio_examples/Singer12_GrindInhale_a.wav', 'audio_examples/Singer12_Effect_PigSqueal.wav', 'audio_examples/Singer14_Effect_DeepGutturals.wav', 'audio_examples/Singer10_Effect_TunnelThroat.wav']

def plot_multi_spectro(x_m, z_m, title_x, title_z, vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', cbar_label="Power (dB)", save=False, y_n_fft=None, sr=48000, n_fft=512):
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
        for j, ax in enumerate(ax_row):
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
                #put a tick every 1000Hz
                idx_tick = int((1000/(1+sr/2)) * (1+n_fft/2))
                tick_locations = np.arange(idx_tick, x_m[0].shape[0], idx_tick)
                ax.set_yticklabels(np.arange(1, 1*len(tick_locations)+1, 1))
                ax.set_yticks(tick_locations)

    fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')

    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    axs[0, 0].set_ylabel(ylabel)
    fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)

    if save:
        plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.savefig('results/spectrograms.pdf')
    plt.show()

if __name__ == "__main__":

    first_mels_list = []
    second_mels_list = []

    idx_max = int((F_MAX/(1+SR/2)) * (1+N_FFT/2))

    for f_name in FIRST_ROW_AUDIOS:

        audio, _ = librosa.load(f_name, sr=SR)
        audio = librosa.util.normalize(audio)[SR:SR*2]
        spec = librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW)
        spec = np.abs(spec)
        first_mels_list.append(spec[:idx_max, :])

    for f_name in SECOND_ROW_AUDIOS:
        audio, _ = librosa.load(f_name, sr=SR)
        audio = librosa.util.normalize(audio)[SR:2*SR]
        spec = librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW)
        spec = np.abs(spec)
        second_mels_list.append(spec[:idx_max, :])

    plot_multi_spectro(first_mels_list, second_mels_list, title_x=['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream'], title_z=['Grind Inhale', 'Pig Squeal', 'Deep Gutturals', 'Tunnel Throat'], ylabel='Frequency (kHz)', y_n_fft=512, vmin=0, vmax=15, cbar_label='Amplitude', sr=SR, n_fft=N_FFT)
