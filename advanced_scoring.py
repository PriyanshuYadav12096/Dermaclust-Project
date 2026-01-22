import pandas as pd
import ast
from ingredients import KEY_INGREDIENTS, COMEDOGENIC_REAGENTS, POTENTIAL_IRRITANTS

# advanced_scoring.py (Updated Version)

def calculate_scientific_score(ingredient_list, skin_type):
    score = 0.0
    # Do NOT use a set() here because order matters for concentration!
    ingredients = [i.strip().lower() for i in ingredient_list]
    
    for index, ingredient in enumerate(ingredients):
        # Weighting factor: Earlier ingredients have much higher impact.
        # This formula gives the 1st ingredient a 1.0 multiplier, 
        # and it decays as you go down the list.
        weight = 1.0 / (index + 1)
        
        # 1. POSITIVE MATCHES (Multiplied by concentration weight)
        if ingredient in KEY_INGREDIENTS.get(skin_type, []):
            score += (2.0 * weight) # High score for active ingredients at the top
            
        # 2. NEGATIVE FILTERS (Heavy penalties if in top 5)
        if skin_type in ['acne', 'oily']:
            if ingredient in COMEDOGENIC_REAGENTS:
                # If a pore-clogger is the 1st ingredient, penalty is -5.0
                # If it's the 10th, penalty is -0.5
                score -= (5.0 * weight) 
                
        if skin_type == 'dry':
            if ingredient in POTENTIAL_IRRITANTS:
                score -= (4.0 * weight)
                
    return max(0, score) # Ensure score doesn't go below 0

# Load your database
df = pd.read_csv("final_product_database.csv")
df['clean_ingreds'] = df['clean_ingreds'].apply(ast.literal_eval)

# Apply scoring for all types
for s_type in ['acne', 'oily', 'dry', 'normal', 'wrinkles']:
    df[f'score_{s_type}'] = df['clean_ingreds'].apply(lambda x: calculate_scientific_score(x, s_type))
    # Normalize score between 0 and 1
    max_val = df[f'score_{s_type}'].max()
    if max_val > 0:
        df[f'score_{s_type}'] = df[f'score_{s_type}'] / max_val

df.to_csv("products_with_scores.csv", index=False)
print("✅ Advanced Database Created with Scientific Weighting!")