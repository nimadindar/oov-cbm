import xml.etree.ElementTree as ET
import os
def load_elvish_dataset():
    """
    Load the Elvish dataset from the Eldamo XML file and return a list of tuples
    containing Elvish words and their corresponding glosses.
    
    Returns:
        List[Tuple[str, str]]: A list of (Elvish word, gloss) tuples.
    """
    # Load XML
    tree = ET.parse(f"{os.getcwd()}/elvish_load/eldamo-data.xml")
    root = tree.getroot()

    # Temporary mappings
    gloss_to_elvish = {}
    elvish_to_gloss = {}

    for word in root.findall(".//word"):
        speech = word.get("speech", "")
        if speech == "phoneme":
            continue

        elvish = word.get("v", "").strip()
        gloss = word.get("gloss", "").strip()

        if not elvish or not gloss:
            continue

        # Pick just the first gloss (e.g. "strength, endurance" → "strength")
        main_gloss = gloss.split(",")[0].strip().lower()

        # Avoid duplicate glosses mapping to many Elvish words
        if main_gloss not in gloss_to_elvish:
            gloss_to_elvish[main_gloss] = elvish
            elvish_to_gloss[elvish] = main_gloss

    # Convert to dataset
    dataset = list(elvish_to_gloss.items())

    print(f"Extracted {len(dataset)} unique word pairs.")
    for elvish, gloss in dataset[:10]:
        print(f"{elvish} -> {gloss}")
    return dataset

