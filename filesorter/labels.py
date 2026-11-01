"""Visual concepts the image model can recognise (zero-shot), and how they map to folders.

Each entry: (description shown to the image model, folder name, top-level area).
The descriptions are English because the image model's text side is English; folder names are short.
"""

from __future__ import annotations

PHOTOS = "Photos & events"
SCREENSHOTS = "Screenshots"
SCANS = "Scans & photographed documents"
GRAPHICS = "Graphics & design"
RECORDINGS = "Screen recordings"
VIDEOS = "Videos"

# Areas whose pictures usually carry readable text (worth reading with OCR and sorting by subject)
TEXTUAL_AREAS = {SCREENSHOTS, SCANS}

LABELS: list[tuple[str, str, str]] = [
    # --- people, events, team life -----------------------------------------------------------------------------
    ("a group photo of colleagues posing together", "Group photos", PHOTOS),
    ("people doing a team building activity", "Team building", PHOTOS),
    ("people bowling in a bowling alley", "Bowling", PHOTOS),
    ("people playing laser tag or paintball", "Laser tag & paintball", PHOTOS),
    ("people in an escape room", "Escape game", PHOTOS),
    ("people karting on a race track", "Karting", PHOTOS),
    ("people cooking together in a cooking class", "Cooking workshop", PHOTOS),
    ("people hiking outdoors in nature", "Hiking & outdoors", PHOTOS),
    ("people on a boat or kayaking", "Boat & water activities", PHOTOS),
    ("people skiing in the snow", "Ski trip", PHOTOS),
    ("people playing a sport together", "Sports", PHOTOS),
    ("a party with drinks and people celebrating", "Party & celebration", PHOTOS),
    ("an office christmas party with decorations", "Christmas party", PHOTOS),
    ("a birthday celebration with a cake", "Birthday", PHOTOS),
    ("a farewell or retirement celebration at the office", "Farewell party", PHOTOS),
    ("colleagues having lunch or dinner at a restaurant", "Team meal", PHOTOS),
    ("a buffet or cocktail reception", "Reception & cocktail", PHOTOS),
    ("a wedding ceremony", "Wedding", PHOTOS),
    ("a speaker giving a talk on stage at a conference", "Conference", PHOTOS),
    ("an award ceremony with people receiving a prize", "Award ceremony", PHOTOS),
    ("a trade show booth or exhibition stand", "Trade show", PHOTOS),
    ("people in a business meeting around a table", "Meetings", PHOTOS),
    ("people in a workshop with sticky notes on a wall", "Workshops", PHOTOS),
    ("people attending a training session in a classroom", "Training sessions", PHOTOS),
    ("a portrait photo of one person", "Portraits", PHOTOS),
    ("a selfie", "Selfies", PHOTOS),
    ("a professional headshot", "Headshots", PHOTOS),
    ("a modern office interior", "Office spaces", PHOTOS),
    ("the outside of a building", "Buildings", PHOTOS),
    ("a construction site", "Construction site", PHOTOS),
    ("an industrial site or factory floor", "Industrial site", PHOTOS),
    ("a city street", "City", PHOTOS),
    ("a landscape with mountains, sea or countryside", "Landscapes", PHOTOS),
    ("a beach", "Beach", PHOTOS),
    ("food on a plate", "Food", PHOTOS),
    ("a dog or a cat", "Pets", PHOTOS),
    ("a car or vehicle", "Vehicles", PHOTOS),
    ("a product photo on a plain background", "Product photos", GRAPHICS),
    ("children playing", "Family", PHOTOS),
    ("a hotel room", "Travel", PHOTOS),
    ("an airport or airplane", "Travel", PHOTOS),
    # --- screenshots ---------------------------------------------------------------------------------------------
    ("a screenshot of a spreadsheet", "Spreadsheets", SCREENSHOTS),
    ("a screenshot of source code in a code editor", "Code", SCREENSHOTS),
    ("a screenshot of a website", "Websites", SCREENSHOTS),
    ("a screenshot of a chat conversation", "Chats & messages", SCREENSHOTS),
    ("a screenshot of an email", "Emails", SCREENSHOTS),
    ("a screenshot of a dashboard with charts", "Dashboards", SCREENSHOTS),
    ("a screenshot of a video call with participants", "Video calls", SCREENSHOTS),
    ("a screenshot of a text document", "Documents", SCREENSHOTS),
    ("a screenshot of a presentation slide", "Slides", SCREENSHOTS),
    ("a screenshot of a software application window", "Software", SCREENSHOTS),
    ("a screenshot of a mobile phone app", "Mobile apps", SCREENSHOTS),
    ("a screenshot of an error message", "Error messages", SCREENSHOTS),
    ("a screenshot of a calendar", "Calendars", SCREENSHOTS),
    ("a screenshot of a map", "Maps", SCREENSHOTS),
    ("a photo of a computer screen", "Photos of screens", SCREENSHOTS),
    # --- documents captured as pictures -------------------------------------------------------------------------
    ("a scanned printed document", "Scanned documents", SCANS),
    ("a receipt", "Receipts", SCANS),
    ("an invoice", "Invoices", SCANS),
    ("a business card", "Business cards", SCANS),
    ("an identity card or passport", "ID documents", SCANS),
    ("a whiteboard with handwritten notes", "Whiteboards", SCANS),
    ("handwritten notes on paper", "Handwritten notes", SCANS),
    ("a printed form filled in by hand", "Forms", SCANS),
    ("a train or plane ticket", "Tickets", SCANS),
    ("a page of a book", "Book pages", SCANS),
    # --- graphics and design -------------------------------------------------------------------------------------
    ("a company logo", "Logos", GRAPHICS),
    ("an icon", "Icons", GRAPHICS),
    ("a poster or flyer design", "Posters & flyers", GRAPHICS),
    ("an infographic", "Infographics", GRAPHICS),
    ("a diagram or flowchart", "Diagrams", GRAPHICS),
    ("an organisation chart", "Org charts", GRAPHICS),
    ("a chart or graph", "Charts", GRAPHICS),
    ("an architectural floor plan", "Floor plans", GRAPHICS),
    ("a technical drawing", "Technical drawings", GRAPHICS),
    ("a user interface mockup or wireframe", "UI mockups", GRAPHICS),
    ("a website banner or social media post", "Social media visuals", GRAPHICS),
    ("an illustration or drawing", "Illustrations", GRAPHICS),
    ("a cartoon or meme", "Memes", GRAPHICS),
    ("a geographic map", "Maps", GRAPHICS),
    ("a presentation slide with text and images", "Slides", GRAPHICS),
    ("a photo with a solid colour background used as a wallpaper", "Backgrounds & wallpapers", GRAPHICS),
    ("a signature", "Signatures", GRAPHICS),
    ("a QR code or barcode", "QR codes", GRAPHICS),
]

LABEL_PROMPTS = [p for p, _, _ in LABELS]

# A photo taken by a camera/phone is a photo of something real: an event, a place, or a document/whiteboard.
CAMERA_AREAS = {PHOTOS, SCANS}


def pick_label(probs, camera: bool = False) -> tuple[int, float, str]:
    """From the image model's probability for every label: best area (sum over its labels), then the best
    label inside that area. Camera photos can only be photos or photographed documents."""
    area_scores: dict[str, float] = {}
    for j, (_, _, area) in enumerate(LABELS):
        if camera and area not in CAMERA_AREAS:
            continue
        area_scores[area] = area_scores.get(area, 0.0) + float(probs[j])
    area = max(area_scores, key=area_scores.get)
    j = max((k for k, lab in enumerate(LABELS) if lab[2] == area), key=lambda k: probs[k])
    return int(j), float(probs[j]), area
