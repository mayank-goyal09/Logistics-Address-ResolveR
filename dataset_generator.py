import pandas as pd
import random
from faker import Faker
from tqdm import tqdm

# Initialize Faker with multiple locales for the Global Multi-Locale Seed
LOCALES = ['en_US', 'en_GB', 'en_IN', 'de_DE', 'fr_FR', 'it_IT', 'es_ES', 'en_AU', 'en_CA', 'nl_NL']
fake = Faker(LOCALES)

# Keyboard proximity map (QWERTY) to simulate human typing errors
PROXIMITY_MAP = {
    'q': 'wa', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'qweadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
    'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn',
    '1': '2q', '2': '13qw', '3': '24we', '4': '35er', '5': '46rt', '6': '57ty', '7': '68yu', '8': '79ui', '9': '80io', '0': '9op'
}

def generate_clean_address():
    """Generates a clean address as a list of (character, label) tuples using diverse global formats."""
    components = []
    
    # Generate raw components
    b_num = fake.building_number()
    street = fake.street_name().lower()
    city = fake.city().lower()
    
    # Not all locales have 'state_abbr' or 'state'. We default to empty string if missing.
    try:
        state = fake.state_abbr().lower()
    except AttributeError:
        try:
            state = fake.state().lower()
        except AttributeError:
            state = ""

    zipcode = fake.postcode()
    
    # Define globally realistic layout templates
    # Labels: 'N': House Number, 'S': Street, 'C': City, 'A': Area/State, 'P': Postal Code, 'O': Other
    formats = [
        # Standard US/CA: Number Street, City, State Zip
        [('N', b_num), ('O', ' '), ('S', street), ('O', ', '), ('C', city), ('O', ', '), ('A', state), ('O', ' '), ('P', zipcode)],
        
        # Standard GER/NL/FR: Street Number, Zip City
        [('S', street), ('O', ' '), ('N', b_num), ('O', ', '), ('P', zipcode), ('O', ' '), ('C', city)],
        
        # Standard UK/AU/IN: Number, Street, City, State, Zip
        [('N', b_num), ('O', ', '), ('S', street), ('O', ', '), ('C', city), ('O', ', '), ('A', state), ('O', ' '), ('P', zipcode)],

        # Simple: Zip Street City
        [('P', zipcode), ('O', ' '), ('S', street), ('O', ' '), ('C', city)],
        
        # Simple string: Number Street City Zip
        [('N', b_num), ('O', ' '), ('S', street), ('O', ' '), ('C', city), ('O', ' '), ('P', zipcode)]
    ]
    
    chosen_format = random.choice(formats)
    
    # Parse format and treat internal spaces in text appropriately
    for label, text in chosen_format:
        if not text:
            continue
            
        for char in str(text):
            # If the layout block is explicitly defined as 'O', it's always 'O' (e.g. ', ')
            if label == 'O':
                components.append((char, 'O'))
            else:
                # If there's a space inside an entity (like "New York"), code it as 'O'
                if char == ' ':
                    components.append((' ', 'O'))
                else:
                    components.append((char, label))
                    
    return components

def apply_perturbations(components):
    """
    Takes a clean list of (character, label) tuples and applies random errors:
    - Merge: Deletes spaces to fuse tokens (Hardest for models).
    - Omission: occasionally deletes normal characters.
    - Transposition: swaps two adjacent characters.
    - Keyboard Proximity: swaps a char with a neighboring key.
    """
    result = []
    i = 0
    while i < len(components):
        char, label = components[i]
        
        error_chance = random.random()
        
        # 1. Merge (Delete a space: ~15% chance for any space) -> e.g., 123 Main St -> 123MainSt
        if char == ' ' and error_chance < 0.15:
            i += 1
            continue
            
        # 2. Omission (Delete a random non-space char: ~3% chance) -> e.g., Main St -> Man St
        elif char != ' ' and error_chance < 0.03:
            i += 1
            continue
            
        # 3. Transposition (Swap with next char: ~4% chance) -> e.g., Main St -> Mian St
        elif error_chance < 0.07 and i < len(components) - 1:
            next_char, next_label = components[i+1]
            if char.isalpha() and next_char.isalpha():
                result.append((next_char, next_label))
                result.append((char, label))
                i += 2
                continue
                
        # 4. Keyboard Proximity Error (~3% chance)
        elif error_chance < 0.10 and char.lower() in PROXIMITY_MAP:
            prox_char = random.choice(PROXIMITY_MAP[char.lower()])
            if char.isupper():
                prox_char = prox_char.upper()
            result.append((prox_char, label))
            i += 1
            continue
            
        # Default: keep character as is
        result.append((char, label))
        i += 1
        
    return result

def create_dataset(num_samples=1000):
    """Generates the full dataset with messy text, tokenized chars, and labels."""
    data = []
    
    for _ in tqdm(range(num_samples), desc="Generating 100k Dateset"):
        clean_components = generate_clean_address()
        messy_components = apply_perturbations(clean_components)
        
        # In case omission removes everything, ensure we have at least something to avoid empty rows
        if not messy_components:
            messy_components = clean_components
            
        raw_input = "".join([c for c, l in messy_components])
        tokenized_chars = [c for c, l in messy_components]
        labels = [l for c, l in messy_components]
        
        data.append({
            "Raw_Input (Messy)": raw_input,
            "Tokenized_Chars": str(tokenized_chars),
            "Labels (Ground Truth)": str(labels)
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate 100,000 rows for the final dataset
    num_rows = 100000 
    print(f"Starting Generation of {num_rows} messy addresses...")
    
    df = create_dataset(num_samples=num_rows)
    
    output_filename = "messy_address_dataset.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\nDataset successfully generated and saved to '{output_filename}'.")
    print(f"Shape of Data: {df.shape}")
    print("\nExample Rows:")
    print(df.head(3))
