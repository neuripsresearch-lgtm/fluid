import nltk
from nltk.corpus import wordnet as wn

# Ensure the WordNet database is downloaded
try:
    wn.ensure_loaded()
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_wnid(word):
    """
    Retrieves the WordNet ID (wnid) for a given word.
    Returns a list of tuples: (wnid, definition)
    """
    synsets = wn.synsets(word)
    
    if not synsets:
        return None

    results = []
    for synset in synsets:
        # Get the Part of Speech (n=noun, v=verb, a=adjective, r=adverb)
        pos = synset.pos()
        
        # Get the offset ID (integer)
        offset = synset.offset()
        
        # Format as ImageNet style wnid: e.g., 'n' + 8-digit-padded-offset
        # Example: n02123045
        wnid = f"{pos}{offset:08d}"
        
        # Get the definition to help distinguish between meanings
        definition = synset.definition()
        
        results.append((wnid, synset.name(), definition))
        
    return results

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("--- Word to WNID Converter ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter a word: ").strip().lower()
        if user_input == 'exit':
            break
            
        results = get_wnid(user_input)
        
        if not results:
            print(f"No WordNet ID found for '{user_input}'. Try a different variation.\n")
        else:
            print(f"\nFound {len(results)} meanings for '{user_input}':")
            print("-" * 60)
            print(f"{'WNID':<15} | {'Synset Name':<20} | {'Definition'}")
            print("-" * 60)
            
            for wnid, name, definition in results:
                print(f"{wnid:<15} | {name:<20} | {definition}")
            print("\n")