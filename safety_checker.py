# safety_checker.py
# This file handles the logic for the "🛡️ Safety Scanner" tab

from ingredients import COMEDOGENIC_REAGENTS, POTENTIAL_IRRITANTS

def check_product_safety(ingredient_text, skin_type):
    """
    Scans a raw text block of ingredients for red flags based on skin type.
    """
    # Convert raw text to a clean list by splitting at commas or new lines
    input_ingreds = [i.strip().lower() for i in ingredient_text.replace(',', '\n').split('\n') if i.strip()]
    
    found_red_flags = []
    
    # 1. Check for Oily/Acne-prone skin irritants (Pore-cloggers)
    if skin_type in ['acne', 'oily']:
        for ing in input_ingreds:
            if ing in COMEDOGENIC_REAGENTS:
                found_red_flags.append({"ingredient": ing, "reason": "Comedogenic (Pore-Clogging)"})
                
    # 2. Check for Dry/Sensitive skin irritants
    if skin_type == 'dry':
        for ing in input_ingreds:
            if ing in POTENTIAL_IRRITANTS:
                found_red_flags.append({"ingredient": ing, "reason": "Potential Irritant/Drying"})
                
    return found_red_flags