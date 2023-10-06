import pandas as pd
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

font = {'family' : 'Times New Roman',
        'size'   : 22}

matplotlib.rc('font', **font)

file_meta = "EMVD/metadata_files.csv"
file_meta_singers = "EMVD/metadata_singers.csv"
df_meta = pd.read_csv(file_meta)
df_meta_singers = pd.read_csv(file_meta_singers)
f_names = df_meta['file_name'].to_numpy()
l_audio_length = []
font_size = 25

#calculate the length of each audio file
for f_name in f_names:
    audio, sr = librosa.load('EMVD/audio/'+f_name, sr=48000)
    audio_len = len(audio)/48000
    l_audio_length.append(audio_len)

f_names_in_folder = []
for subdir, dirs, files in os.walk('CTED/'):
    for file in files:
        f = os.path.join(subdir, file)
        f_names_in_folder.append(file)

#check if there are any differences between the names in the audio folder and the name in the metadatas
differences = [item for item in f_names_in_folder if item not in f_names]

df_meta['audio_len'] = l_audio_length
df_meta = df_meta[df_meta['type']!='Other']

df_meta = df_meta[df_meta['authors_rank']!='C']
df_meta = df_meta[df_meta['authors_rank']!='-']

print('Total duration')
print(np.sum(df_meta['audio_len'].to_numpy()))

print('Clear Voice duration')
print(np.sum(df_meta[df_meta['name']=='ClearVoice']['audio_len'].to_numpy()))

df_bar = df_meta.groupby(['name', 'type']).agg(total_audio_len=('audio_len', 'sum'),
                                    count_audio=('audio_len', 'count')).reset_index()
df_bar_perc = df_bar.copy()
df_bar_perc['total_audio_len'] = 100 * df_bar_perc['total_audio_len'] / df_bar_perc['total_audio_len'].sum()

print('DURATION OF EACH TECHNIQUE')
print(df_bar_perc)


#########################
## PLOT THE DURATION ANALYSIS FROM THE PAPER, AND SAVE IT IN 'results'

col_pal = sns.color_palette("colorblind")
order = ['ClearVoice', 'BlackShriek', 'DeathGrowl', 'HardcoreScream', 'GrindInhale', 'PigSqueal', 'DeepGutturals', 'TunnelThroat']
short_name = ['CV', 'BS', 'DG', 'HS', 'GI', 'PS', 'DeG', 'TT']
full_name = ['Clear Voice', 'Black Shriek', 'Death Growl', 'Hardcore Scream', 'Grind Inhale', 'Pig Squeel', 'Deep Gutturals', 'Tunnel Throat']
colors = [col_pal[9], col_pal[9], col_pal[9], col_pal[9], col_pal[9], col_pal[5], col_pal[5], col_pal[5]]

df_bar_singer = df_meta.groupby('name')['singer_id'].nunique().reset_index()
df_bar_singer.columns = ['name', 'num_singers']

# Create a figure with two subplots, one for each bar graph
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Bar graph 1 - Duration of each vocal technique and vocal effect
ax1 = axs[0]

# Create a bar plot for duration
ax1 = sns.barplot(data=df_bar, y='name', x='total_audio_len', palette=colors, order=order, ax=ax1)
ax1.set_yticklabels(full_name)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xlabel('Samples total duration (s)')
ax1.set_ylabel('')

# Bar graph 2 - Number of singers for each technique
ax2 = axs[1]

# Create a bar plot for the number of singers
ax2 = sns.barplot(data=df_bar_singer, y='name', x='num_singers', palette=colors, order=order, ax=ax2)
ax2.set_yticklabels(full_name)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xlabel('Number of singers')
ax2.set_ylabel('')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('results/dataset_duration_and_number_of_singers.pdf')