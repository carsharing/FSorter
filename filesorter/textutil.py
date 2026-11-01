"""Text helpers: tokenizing, stop words (many languages), keyword phrases and safe folder names."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Compact stop-word lists. They only need to remove words that would make bad folder names.
_STOP = {
    "en": """a about above after again against all also am an and any are aren as at be because been before being
        below between both but by can cannot could did do does doing down during each few for from further had has
        have having he her here hers herself him himself his how i if in into is it its itself just let me more most
        my myself no nor not now of off on once only or other ought our ours ourselves out over own same she should
        so some such than that the their theirs them themselves then there these they this those through to too under
        until up very was we were what when where which while who whom why will with would you your yours yourself
        yourselves per via etc may might must shall one two three new use used using get got also within without
        across among upon please thanks thank regards dear hello hi best kind""",
    "fr": """a à afin ai aie aient aies ait alors as au aucun aucune aupres auquel aura aurai auraient aurais aurait
        auras aurez auriez aurions aurons auront aussi autre autres aux auxquelles auxquels avaient avais avait avant
        avec avez aviez avions avoir avons ayant ayez ayons bon c ça car ce ceci cela celle celles celui cependant ces
        cet cette ceux chacun chaque chez ci comme comment d dans de des desquelles desquels deux devrait dois doit
        donc dont du duquel elle elles en encore entre es est et étaient étais était étant été êtes étiez étions être
        eu eue eues eûmes eurent eus eut eux fait faire fais faut fois font furent fus fut ici il ils j je jusqu
        jusque l la là laquelle le lequel les lesquelles lesquels leur leurs lors lorsque lui m ma mais me même mêmes
        mes moi moins mon n ne ni non nos notre nôtre nous on ont ou où par parce pas peu peut peuvent plus pour
        pourquoi qu quand que quel quelle quelles quels qui quoi s sa sans se selon sera serai seraient serais serait
        seras serez seriez serions serons seront ses si sien siens soi soient sois soit sommes son sont sous soyez
        soyons suis sur t ta te tes toi ton tous tout toute toutes très tu un une unes uns vers voici voilà vos votre
        vôtre vous vu y cordialement bonjour bonsoir merci madame monsieur mme mlle salut bien bonne journée""",
    "es": """a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante e el ella
        ellas ellos en entre era es esa esas ese eso esos esta estaba estado estas este esto estos fue fueron ha han
        hasta hay la las le les lo los mas más me mi mis mucho muy nada ni no nos nosotros o otra otros para pero poco
        por porque que qué quien se ser si sí sin sobre su sus también tan te tiene todo todos tu tus un una uno unos
        y ya yo gracias hola saludos""",
    "de": """aber alle allem allen aller alles als also am an andere anderen auch auf aus bei bin bis bist da damit
        dann das dass dein deine dem den der des dich die dies diese diesem diesen dieser dieses dir doch dort du
        durch ein eine einem einen einer eines er es euer eure für hab habe haben hat hatte hier hin hinter ich ihm
        ihn ihr ihre im in ist ja jede jedem jeden jeder jedes jetzt kann kein keine mich mein meine mit muss nach
        nicht nichts noch nun nur ob oder ohne sehr sein seine sich sie sind so über um und uns unser unter vom von
        vor war waren was weil welche wenn wer werden wie wieder wir wird zu zum zur danke hallo grüße""",
    "it": """a ad al alla alle anche avere che chi ci come con cosa da dal dalla dei del della delle di e è ed gli
        ha hanno i il in io la le lei lo loro lui ma mi mio ne nel nella noi non o per perché più quale quando
        quella quelle quello questa queste questo se si sono su sua sue suo tra tu tutti tutto un una uno voi
        grazie ciao saluti""",
    "pt": """a ao aos as até com como da das de dela dele do dos e é ela elas ele eles em entre era essa esse esta
        este eu foi há isso isto já lhe mais mas me mesmo meu minha muito na nas não nem no nos nós o os ou para
        pela pelo por qual quando que quem se sem ser seu sua são também te tem um uma você obrigado olá""",
    "nl": """aan al alles als bij dan dat de der deze die dit doch door een en er had heb hebben het hier hij hoe
        hun ik in is je kan maar me met mij na naar niet niets nog nu of om ook op over te tot u uit van veel voor
        was wat we wel werd wie wij wil zal ze zelf zich zij zijn zo zou bedankt hallo groeten""",
}

# Words that appear in file names, e-mails and office files but never describe a subject.
_NOISE = """pdf doc docx xls xlsx xlsm ppt pptx txt csv rtf odt ods odp msg eml jpg jpeg png gif bmp tif tiff webp heic
    mp4 mov avi mkv wmv mp3 wav zip rar htm html xml json img dsc dscn dcim pxl image images photo photos picture
    pictures file files fichier fichiers document documents dossier folder new nouveau nouvelle untitled sans titre
    copy copie kopie copia final finale version draft brouillon scan scanned scanne numérisation export exported
    temp tmp test page pages sheet feuil feuille sheet1 feuil1 slide diapositive de of mail email e-mail courriel
    from sent subject objet envoyé envoye cc bcc cci re fw fwd tr to date pièce jointe attachment attachments
    http https www com fr org net html outlook gmail hotmail mailto click cliquez here ici unsubscribe désabonner
    mon tue wed thu fri sat sun lun mar mer jeu ven sam dim monday tuesday wednesday thursday friday saturday sunday
    lundi mardi mercredi jeudi vendredi samedi dimanche january february march april june july august september
    october november december janvier février fevrier avril juin juillet août aout septembre octobre novembre
    décembre decembre whatsapp screenshot capture écran ecran recording enregistrement video vidéo clip media
    download downloads téléchargement téléchargements telechargements desktop bureau misc divers other autre
    autres stuff truc data donnees données info infos information informations total sous yes oui none null true
    false nan
    find attached attach kindly regards dear thanks thank reminder remind noticed following discussed discuss
    explains explain summary regarding asap okay sure great also still just like let know need needs want wants
    make made take see look looking forward hope hoping please could would should will shall can get got send
    sent sending attached enclosed below above mentioned ensure call calls meeting meetings
    veuillez trouver trouverez joint jointe ci-joint convenu convenue rappel rappeler noter constaté constate
    constatons suite échange echange échanges précise precise préciser conclusion concernant suivant suivante
    suivants important importante importants merci vouloir cordialement souhaite souhaitons faire fait faites
    voir prendre mettre besoin pouvez pourriez pouvons envoyer envoie transmettre transmets retour rapidement
    possible ensemble cela ceci voici ainsi afin également egalement notamment donc toutefois cependant
    bonjour bonsoir madame monsieur salutations sincères""".split()

STOPWORDS: frozenset[str] = frozenset(
    w for words in _STOP.values() for w in words.split()
) | frozenset(_NOISE)

# Small words allowed *inside* a keyword phrase ("note de frais", "plan of action").
_GLUE = frozenset("de du des d la le l en à au aux of the for in on del di da do dos das van von der".split())

# Folder names that say nothing about the content (used when reading path context).
GENERIC_FOLDERS = frozenset(
    s.strip().lower()
    for s in """desktop, bureau, documents, mes documents, my documents, downloads, téléchargements,
    telechargements, pictures, images, mes images, my pictures, photos, videos, vidéos, mes vidéos, my videos,
    music, musique, dcim, camera, camera roll, pellicule, screenshots, screen shots, captures, captures d’écran,
    captures d'écran, captures d ecran, new folder, nouveau dossier, misc, divers, temp, tmp, onedrive, dropbox,
    google drive, icloud drive, users, utilisateurs, public, home, documents partagés, shared, attachments,
    pièces jointes, whatsapp, whatsapp images, whatsapp video, whatsapp documents, telegram, telegram desktop,
    100apple, 101apple, 100andro, 100media, sorted, unsorted, archive, archives, old, ancien, backup, sauvegarde,
    save, export, exports, scans, scan, files, fichiers, data, new, nouveau, a trier, à trier, to sort, inbox,
    work, travail, perso, personal, personnel, other, autres, stuff, recordings, enregistrements""".split(",")
    if s.strip()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CAMEL_RE = re.compile(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ])")
_SPACES_RE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    """Lower-cased alphabetic words (any script). Digits and punctuation are separators."""
    return _WORD_RE.findall(text.lower())


def is_meaningful(word: str) -> bool:
    if len(word) < 3 or len(word) > 30 or word in STOPWORDS:
        return False
    # Words without vowels are usually codes or abbreviations ("xyzt", "rdv" is fine though).
    if len(word) > 4 and not re.search(r"[aeiouyàâäéèêëîïôöùûüœæáíóúãõåøаеиоуыэюяαεηιουω]", word):
        return False
    return True


def keywords(text: str) -> list[str]:
    return [w for w in words(text) if is_meaningful(w)]


def split_name(name: str) -> str:
    """'RapportFinal_2024-v2.docx' -> 'Rapport Final 2024 v2'."""
    stem = name.rsplit(".", 1)[0] if "." in name[1:] else name
    stem = _CAMEL_RE.sub(" ", stem)
    stem = re.sub(r"[_\-.+()\[\]{}#~=,;]+", " ", stem)
    return normalize_spaces(stem)


def terms_and_phrases(text: str, max_words: int = 4000) -> set[str]:
    """Distinct keywords plus 2-3 word phrases ('note de frais', 'vehicle routing') found in the text."""
    toks = words(text)[:max_words]
    out: set[str] = set()
    n = len(toks)
    for i, t in enumerate(toks):
        if not is_meaningful(t):
            continue
        out.add(t)
        # Bigram: meaningful + meaningful
        if i + 1 < n and is_meaningful(toks[i + 1]) and toks[i + 1] != t:
            out.add(f"{t} {toks[i + 1]}")
        # Trigram with a small glue word in the middle: "note de frais"
        if i + 2 < n and toks[i + 1] in _GLUE and is_meaningful(toks[i + 2]) and toks[i + 2] != t:
            out.add(f"{t} {toks[i + 1]} {toks[i + 2]}")
    return out


def surface_form(phrase: str, sample: str) -> str:
    """How a lower-cased keyword is actually written in the documents ('vpn access' -> 'VPN access',
    'chiffre d affaires' -> "chiffre d'affaires")."""
    parts = phrase.split(" ")
    pattern = r"(?<!\w)" + r"[\s'’\-]+".join(re.escape(p) for p in parts) + r"(?!\w)"
    found = re.findall(pattern, sample, flags=re.IGNORECASE)
    if not found:
        return phrase
    best = Counter(f for f in found).most_common()
    # prefer the most common form; on ties, prefer one that keeps an acronym in capitals
    form = max(best, key=lambda kv: (kv[1], sum(c.isupper() for c in kv[0][1:])))[0]
    return normalize_spaces(form.replace("’", "'"))


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def nice_case(phrase: str) -> str:
    parts = phrase.split(" ")
    res = []
    for i, p in enumerate(parts):
        if i > 0 and p in _GLUE:
            res.append(p)
        elif len(p) <= 3 and p.isalpha() and i > 0 and p not in STOPWORDS and len(parts) > 1 and p.upper() == p:
            res.append(p)
        else:
            res.append(p[:1].upper() + p[1:])
    return " ".join(res)


# ---------------------------------------------------------------------------------------------------------------
# Windows-safe folder and file names
# ---------------------------------------------------------------------------------------------------------------
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name: str, max_len: int = 60, fallback: str = "Untitled") -> str:
    s = _INVALID_CHARS.sub(" ", name or "")
    s = normalize_spaces(s).strip(" .")
    if not s:
        s = fallback
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    if len(s) > max_len:
        cut = s[:max_len]
        if " " in cut[max_len // 2:]:
            cut = cut[: cut.rfind(" ")]
        s = cut.strip(" .,&-")
    return s or fallback


def unique_name(name: str, taken: set[str]) -> str:
    """Return name, or 'name (2)', 'name (3)', ... so it is not in `taken` (case-insensitive, like Windows)."""
    lowered = {t.lower() for t in taken}
    if name.lower() not in lowered:
        return name
    i = 2
    while f"{name} ({i})".lower() in lowered:
        i += 1
    return f"{name} ({i})"


_COPY_MARKER = re.compile(
    r"(\(\d+\)|\bcopy\b|\bcopie\b|\bkopie\b|\bcopia\b|\bduplicate\b|\bdupliqu\w*|\bcopy of\b|\s-\s\d+$)",
    re.IGNORECASE,
)


def looks_like_copy(filename: str) -> bool:
    stem = filename.rsplit(".", 1)[0]
    return bool(_COPY_MARKER.search(stem))
