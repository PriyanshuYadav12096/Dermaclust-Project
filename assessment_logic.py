# assessment_logic.py

def calculate_hybrid_profile(ai_results, quiz_answers):
    """
    ai_results: Dict from CNN (e.g., {'oily': 0.8, 'dry': 0.1 ...})
    quiz_answers: Dict from User (e.g., {'feel_after_wash': 'tight', 'pores': 'visible'})
    """
    
    # Weighting: 40% AI, 60% Quiz (Quiz is more reliable for skin behavior)
    weights = {
        'acne': 0.0,
        'dry': 0.0,
        'oily': 0.0,
        'normal': 0.0,
        'wrinkles': 0.0
    }

    # Add AI weights
    for skin_type, confidence in ai_results.items():
        weights[skin_type] += (confidence * 0.4)

    # Add Quiz weights based on clinical symptoms
    if quiz_answers.get('feel_after_wash') == 'Tight/Itchy':
        weights['dry'] += 0.3
    elif quiz_answers.get('feel_after_wash') == 'Greasy/Shiny':
        weights['oily'] += 0.3
        
    if quiz_answers.get('breakouts') == 'Frequent':
        weights['acne'] += 0.3
        
    if quiz_answers.get('concerns') == 'Fine lines':
        weights['wrinkles'] += 0.3

    # Return the type with the highest combined weight
    final_type = max(weights, key=weights.get)
    return final_type, weights