from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample transaction dataset
data = {
    'Transaction_ID': [1, 1, 2, 2, 3, 4, 4, 5],
    'Item': ['Milk', 'Bread', 'Bread', 'Diapers', 'Milk', 'Bread', 'Diapers', 'Milk']
}

df = pd.DataFrame(data)

# Convert the transactional dataset into a one-hot encoded DataFrame
onehot = pd.get_dummies(df['Item'])

# Apply the Apriori algorithm to find frequent itemsets with a lower minimum support
frequent_itemsets = apriori(onehot, min_support=0.2, use_colnames=True)

# Generate association rules with a minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the resulting association rules
print(rules)
