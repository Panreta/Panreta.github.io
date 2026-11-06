---
layout: post
title: 'Extracting music from game'
date: 2026-11-06 11:03
description: "The origin of care divergence and why it has such property."
tags: [Software Development, Game]
categories: AI for fun
tabs: true
---

# Extracting Music from Servant of the Lake (Unity Game) on macOS

This manual extracts the music from the Steam game *Servant of the Lake* (Rusty Lake) and converts it to MP3. The same method works for most Unity games that use Unity's built-in audio.

The extracted files are for personal listening. The official soundtracks are sold on Victor Butzelaar's Bandcamp (victorbutzelaar.bandcamp.com), with proper track names and full album mixes.

---

## Overview

The game's music is already on disk, but not as MP3 files. Unity packs all its audio into its own file format. The Python library **UnityPy** can read that format and save every audio clip as a WAV file. You then pick out the music tracks and convert them to MP3 with **ffmpeg**.

How the game stores its audio:

| File | What it holds |
| --- | --- |
| `data.unity3d` (~315 MB) | A compressed bundle with the index: each `AudioClip`'s name, format, and where its audio data sits |
| `sharedassets0.resource` – `sharedassets16.resource` | The raw audio data the index points into |
| `StreamingAssets/AudioVolumes.csv` | Volume settings only, no music |

The `.resource` files can't be extracted alone. Without the index in `data.unity3d`, the tool doesn't know where one clip ends and the next begins. That's why the script loads the whole data folder at once.

If a game has `.bank` files instead, it uses the FMOD audio engine, and this method does not apply.

---

## 1. Locate the game data folder

In Steam, right-click **Servant of the Lake** → **Manage** → **Browse local files**. Then right-click `ServantOfTheLake.app` → **Show Package Contents** and open `Contents/Resources/Data/`.

The full path is:

```
/Users/sanchez/Library/Application Support/Steam/steamapps/common/Servant of the Lake/ServantOfTheLake.app/Contents/Resources/Data
```

Optional check: to confirm the game uses Unity's built-in audio, open a terminal in that folder and run:

```bash
find . -name "*.resource" -o -name "*.assets" -o -name "*.bank"
```

This searches the folder and its subfolders for files with those extensions. `.resource` / `.assets` files mean Unity's built-in audio (this manual applies). `.bank` files mean FMOD (this manual does not apply).

---

## 2. Set up a Python environment

Homebrew's Python refuses `pip install` (the `externally-managed-environment` error), so use a dedicated conda environment.

Create the environment:

```bash
conda create -n unity-extract python=3.12 -y
```

This creates an environment named `unity-extract` with Python 3.12, stored in `~/miniconda3/envs/unity-extract`. Python 3.12 is used because newer versions (such as 3.14) may not yet have prebuilt packages for UnityPy's dependencies. `-y` skips the confirmation prompt.

Activate it:

```bash
conda activate unity-extract
```

This switches the terminal to that environment, so `python` and `pip` now refer to it. The prompt should start with `(unity-extract)`.

If you get an error saying conda isn't initialized, run:

```bash
conda init zsh
```

This adds conda's setup code to `~/.zshrc`. Close the terminal, open a new one, and activate again.

Install UnityPy:

```bash
python -m pip install UnityPy
```

`python -m pip` runs pip through the active `python`, so the package is guaranteed to go into this environment.

In VS Code, press `Cmd+Shift+P` → **Python: Select Interpreter** → choose `unity-extract`, so VS Code uses the same environment.

---

## 3. Create a work folder

```bash
mkdir -p ~/rusty-lake-music && cd ~/rusty-lake-music
```

`mkdir -p` creates the folder. `-p` means "parents": it creates any missing parent folders and doesn't fail if the folder already exists. `cd` moves into it.

Don't run the script from `/` (the root of the disk). The script writes its output into the current folder, and `/` isn't writable without admin permission.

---

## 4. Write the extraction script

Save this as `~/rusty-lake-music/extract_audio.py`:

```python
import os
import sys
import UnityPy

src = sys.argv[1]                      # game data folder, passed on the command line
out = "extracted_audio"
os.makedirs(out, exist_ok=True)

env = UnityPy.load(src)                # loads every Unity file in the folder
for obj in env.objects:                # walks through every object the game stores
    if obj.type.name == "AudioClip":   # keep only audio clips
        clip = obj.read()
        for name, data in clip.samples.items():   # decodes the clip into WAV bytes
            with open(os.path.join(out, name), "wb") as f:
                f.write(data)
            print("saved", name)
```

How it works:

- `sys.argv` is the list of words on the command line. `sys.argv[1]` is the first argument after the script name: the game's data folder.
- `os.makedirs(out, exist_ok=True)` creates `extracted_audio/`, without failing if it already exists.
- `UnityPy.load(src)` opens every Unity file in the folder, so clips in `data.unity3d` can find their data in the `.resource` files.
- A Unity game stores many object types (textures, meshes, scripts, audio). The `if` keeps only `AudioClip` objects.
- `clip.samples` decodes each clip's compressed audio into WAV data, as a dictionary of `{file name: bytes}`. Each entry is written to `extracted_audio/`.

---

## 5. Run the script

With `(unity-extract)` in the prompt, from `~/rusty-lake-music`:

```bash
python extract_audio.py "/Users/sanchez/Library/Application Support/Steam/steamapps/common/Servant of the Lake/ServantOfTheLake.app/Contents/Resources/Data"
```

This runs the script on the game's data folder. The quotes are required because the path contains spaces ("Application Support", "Servant of the Lake"). Without them, the path would be split into several arguments.

With the path, `sys.argv` looks like:

```python
["extract_audio.py", "/Users/sanchez/Library/Application Support/.../Data"]
```

The script prints a `saved ...` line for each clip. It takes a minute or two.

---

## 6. Separate the music

`extracted_audio/` now holds over a thousand files: sound effects (`Tile_Select-008.wav`), voice lines (`aldous_knife.wav`), ambience loops (`AMB_Crypt.wav`), and music. Every music track is named after the composer: `Victor Butzelaar - Rusty Lake Servant - ...`. A few use the typo `Vicor Butzelaar`.

```bash
cd ~/rusty-lake-music
mkdir -p music
mv extracted_audio/*Butzelaar* music/
```

`mkdir -p music` creates the `music` folder. `mv` moves files, and the pattern `*Butzelaar*` matches any file name containing "Butzelaar" anywhere, which catches both spellings.

Notes on what you get:

- Full pieces include `Main Theme (variation 1 Short)`, `Dreamy Theme (Piano Strings Verlengd)`, `Cooking Theme`, `Alchemy Theme`, `Serving Dinner (Part 1–3)`, `Walking the Closing Patrol`, and `Where is the cat`.
- "Verlengd" is Dutch for "extended". "Laag" means "layer": the cave music is built from layers stacked during gameplay, so `The Cave Laag 1-4 (Compleet)` is the full mix.
- `Altfluit Effect` and `Harp Effect` files are short stingers, not full tracks.
- A few files outside the naming pattern may also be music or singing, such as `Cave_Song1`, `Cave_Song2`, and `Suppertime`. Listen to them before deleting `extracted_audio/`.
- Some sound-effect files share a name (e.g. `prep_dining.wav` three times), so later ones overwrote earlier ones. No music track is duplicated, so the music is complete.

---

## 7. Convert to MP3

Install ffmpeg (once):

```bash
brew install ffmpeg
```

This installs ffmpeg, a command-line audio/video converter, through Homebrew.

Convert every WAV in `music/`:

```bash
for f in music/*.wav; do ffmpeg -i "$f" -b:a 192k "${f%.wav}.mp3"; done
```

- `for f in music/*.wav; do ...; done` runs the command once for each WAV file, with `$f` holding its name.
- `-i "$f"` is the input file.
- `-b:a 192k` sets the MP3 audio bitrate to 192 kbps (good quality, about 1/7 the size of the WAV).
- `${f%.wav}` removes `.wav` from the end of the name, so `Cooking Theme.wav` becomes `Cooking Theme.mp3`.
- The quotes keep names with spaces together.

In ffmpeg's output, `Duration` on the input should match `time=` on the output, which confirms the whole track was converted.

---

## 8. Verify and clean up

Check that every WAV got an MP3:

```bash
ls music/*.wav | wc -l; ls music/*.mp3 | wc -l
```

`ls` lists the matching files and `wc -l` counts the lines, so this prints the WAV count, then the MP3 count. If they match, delete the WAVs:

```bash
rm music/*.wav
```

`rm` deletes files permanently (they don't go to the Trash), so run it only after the counts match.

The finished MP3s are in `~/rusty-lake-music/music/`.

---

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `error: externally-managed-environment` | `pip` ran with Homebrew's Python, not the conda environment | Run `conda activate unity-extract`, check that `which pip` shows `.../miniconda3/envs/unity-extract/bin/pip`, then use `python -m pip install UnityPy` |
| `Run 'conda init' before 'conda activate'` | conda isn't set up in zsh | Run `conda init zsh`, then open a new terminal |
| `IndexError: list index out of range` at `sys.argv[1]` | The script was run without the folder path | Add the quoted data-folder path after `extract_audio.py` |
| `Permission denied` when saving | The terminal is in `/` or another protected folder | `cd ~/rusty-lake-music` first |
| No music files, but `.bank` files exist | The game uses FMOD, not Unity audio | This method doesn't apply; FMOD needs a different extraction tool |
| `zsh: command not found: ffmpeg` | ffmpeg isn't installed | `brew install ffmpeg` |
