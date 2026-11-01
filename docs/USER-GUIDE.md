# Local File Sorter – user guide

Local File Sorter reads what is inside your files and **copies** them into precise folders. For example:

```
Sorted files\
├── Finance\
│   ├── Notes de frais\                      ← expense reports, receipts, related e-mails…
│   │   └── duplicate\                       ← extra copies of the same content
│   └── Budget prévisionnel 2025\
├── Training\
│   └── Formation Excel – tableaux croisés\
├── Photos & events\
│   ├── 2024-06-14 Séminaire Annecy\         ← photos & videos taken at the same event
│   └── 2024-12-19 Christmas party\
├── Screenshots\Spreadsheets\
├── _Unsorted\                               ← files it could not understand (by type)
└── Sorting report.html                      ← where every file went, and why
```

**Your original files are never moved, renamed or deleted.** Everything is copied.

**Nothing leaves your computer.** All the analysis is done by models stored in the app's `models` folder. The app
has no internet code, and it blocks every network connection it might make. Turning off Wi-Fi changes nothing
for it.

## Install (no administrator rights needed)

1. Download `LocalFileSorter-windows.zip`. For nicer folder names, also download
   `LocalFileSorter-naming-model.zip`.
2. Unzip both into the same place, for example `C:\Users\<you>\AppData\Local\Programs\`. You should get
   `LocalFileSorter\LocalFileSorter.exe` and `LocalFileSorter\models\…`.
3. Double-click `LocalFileSorter.exe`. If Windows SmartScreen says "Windows protected your PC", click
   **More info → Run anyway**. The app isn't signed with a paid certificate, which is why this appears.

## Use

1. **Folders to sort**: add one or more folders, for example Documents, Downloads, Desktop and Pictures.
2. **Where to put the sorted copies**: choose an empty folder, ideally on a drive with enough free space. The
   window shows how much space the copies need.
3. **Options**:
   - *Folder detail*: broad, balanced or very precise folders.
   - *OCR*: reads the text in screenshots, scans and screen recordings, so they are sorted by subject.
   - *Name folders with the local AI model*: available when the naming model is installed.
   - *Keep sorting new files automatically*: background mode, see below.
   - *Start automatically when I sign in to Windows*.
4. Click **Start sorting**. You can close the window: with background mode on, the app stays in the system tray
   (bottom-right, near the clock).

When it finishes, click **Open report** to see every file, the folder it went to and why. You can search,
filter (sorted / duplicates / unsorted / problems) and export a CSV.

### The first run takes a while

The first run reads every file. As a rough guide, on a normal office laptop:

| Content | Time |
|---|---|
| 10,000 documents | about 30–60 minutes |
| 10,000 photos | about 20–40 minutes |
| 100 GB of mixed files | a few hours, depending on how many are documents |

Copying itself is limited by disk speed. You can keep working: the app runs at low priority. If you stop it, the
next run continues where it stopped.

### Background mode

After the first run the folder structure is fixed, and the app watches your folders:

- A new file about an existing subject is copied into that subject's folder.
- A few new files about a new subject get a **new folder**.
- New photos join the event folder of the same day, or create a new event folder.
- A file that changes is copied again, in the same place.
- A file whose content already exists (even under another name) goes into that folder's `duplicate` subfolder.

It never asks you anything while it works. Use the tray icon to pause, sort now, open the report or quit.

## Good to know

- **Duplicates** are found by content: identical files, the same document saved under another name or format,
  and the same picture resized or recompressed. The best copy goes into the folder (highest resolution, the
  original name rather than "Copy of…" or "(2)"), and the others go into `duplicate`.
- **`_Unsorted`** holds files that couldn't be understood: programs and installers, unreadable or
  password-protected files, files with no text, and documents whose subject no other file shares. It is
  grouped by file type.
- **OneDrive "online-only" files** (cloud icon) are skipped, because reading them would download them. Make them
  "Always keep on this device" if you want them sorted.
- **Outlook mailboxes** (`.pst`/`.ost`) aren't read. E-mails saved as `.msg` or `.eml` files are.
- Hidden and system folders, and program folders like `C:\Windows`, are never scanned.
- To **start over** with a different structure, choose a new, empty destination folder. The app's memory
  (`_FileSorter` folder) lives inside the destination.

## Where things are

| What | Where |
|---|---|
| Settings | `%LOCALAPPDATA%\LocalFileSorter\settings.json` |
| Log file | `%LOCALAPPDATA%\LocalFileSorter\logs\filesorter.log` |
| Memory of what was sorted | `<destination>\_FileSorter\state.sqlite` |
| Previous reports | `<destination>\_FileSorter\reports\` |

## Command line (optional)

```
LocalFileSorter.exe run --source "C:\Users\me\Documents" --source "C:\Users\me\Downloads" --dest "D:\Sorted"
LocalFileSorter.exe watch --use-saved          (keep sorting, without a window)
LocalFileSorter.exe selftest --out result.json (checks models, OCR and the offline guard)
```
