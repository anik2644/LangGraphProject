from datasets import Dataset

people = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35},
    {"name": "Diana", "age": 28}
]

print("Original list:")
for person in people:
    print(person)

def format_person(persona):
    return {
        "description": f"{persona['name']} is {persona['age']} years old",
        "is_adult": persona['age'] >= 18
    }

# Convert to Hugging Face Dataset
peopleww = Dataset.from_list(people)
print("------------------------------------")

# Apply map to the dataset
result_dataset = peopleww.map(format_person)
print(list(result_dataset))