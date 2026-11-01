Sorts the files on your PC into precise subject folders by reading what's inside them. It copies them; your originals are never moved or changed. Everything runs on your computer and the app blocks all internet access.

## Install (no administrator rights needed)

1. Download both files below:
   - **LocalFileSorter-windows.zip**: the app with its text and image models.
   - **LocalFileSorter-naming-model.zip**: the optional local AI model that gives folders natural names. Without it, folders are named from keywords.
2. Unzip both **into the same folder**, for example `C:\Users\<you>\AppData\Local\Programs\`. You should end up with `LocalFileSorter\LocalFileSorter.exe` and `LocalFileSorter\models\…`.
3. Run `LocalFileSorter.exe`. If Windows SmartScreen appears, click **More info → Run anyway**. The app isn't signed with a paid certificate, which is why this appears.

Installed size is about 2.5 GB. Because files are copied, the destination drive also needs about as much free space as the files you sort.

## First use

Try it on a **copy of one folder** first, with an empty destination folder. Then open `Sorting report.html` in the destination to see where every file went and why.

The full guide is `USER-GUIDE.md`, inside the app folder.

## Known limitations

- Folder and picture labels can be imperfect, especially for screen recordings and events.
- Outlook mailboxes (`.pst`/`.ost`) aren't read; e-mails saved as `.msg`/`.eml` are.
- OneDrive "online-only" files are skipped, because reading them would download them.
- The first run over a large collection (tens of GB) can take hours. It runs at low priority, and if stopped, the next run continues where it left off.
